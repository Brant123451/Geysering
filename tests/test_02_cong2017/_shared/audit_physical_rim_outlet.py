#!/usr/bin/env python3
"""Audit resolved liquid crossing of the Campaign-2 physical riser rim.

The auditor is deliberately independent of case names and experimental
labels.  It consumes VTK PolyData sampled just above the *physical* riser rim
and reports a conservative, mesh-resolution-based crossing decision.

The companion ``sample_physical_rim_readonly.sh`` creates the samples in a
scratch case.  It symlinks source fields read-only by convention and directs
all post-processing output to the scratch tree; no solver is started.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = 2
ALPHA_INTERFACE = 0.5
RIM_SAMPLE_OFFSET_M = 1.0e-7
AREA_SUPPORT_FRACTION = 0.95
# OpenFOAM's XML VTK writer stores TimeValue as Float32 even when directory
# names retain decimal write times.  This tolerance exceeds the Float32 ULP at
# 20 s but remains orders of magnitude below the 0.05 s write interval.
TIME_MATCH_ATOL_S = 2.5e-6

_NUMERIC_DIR = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d+)?$")
_TIME_LINE = re.compile(
    r"^Time\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_TRUE_FATAL = re.compile(
    r"FOAM FATAL|Floating point exception|Segmentation fault|\bNaN\b|"
    r"MPI_ABORT|abnormal exit",
    re.IGNORECASE,
)

_VTK_DTYPES = {
    "Float32": np.dtype("f4"),
    "Float64": np.dtype("f8"),
    "Int8": np.dtype("i1"),
    "UInt8": np.dtype("u1"),
    "Int16": np.dtype("i2"),
    "UInt16": np.dtype("u2"),
    "Int32": np.dtype("i4"),
    "UInt32": np.dtype("u4"),
    "Int64": np.dtype("i8"),
    "UInt64": np.dtype("u8"),
}


class AuditInputError(RuntimeError):
    """Raised when source evidence is absent, inconsistent, or unsupported."""


@dataclass(frozen=True)
class SurfaceFrame:
    time_s: float
    source_time_directory: str
    path: Path
    sha256: str
    plane_z_min_m: float
    plane_z_max_m: float
    opening_area_m2: float
    n_opening_elements: int
    min_opening_element_area_m2: float | None
    max_alpha: float | None
    max_positive_alpha_uz_m_s: float | None
    positive_alpha_weighted_flow_m3_s: float | None
    resolved_upward_area_m2: float | None
    largest_resolved_component_area_m2: float | None
    alpha_below_zero_count: int | None
    alpha_above_one_count: int | None
    data_basis: str
    missing: tuple[str, ...]


def _native_dtype(vtk_type: str, byte_order: str) -> np.dtype:
    try:
        dtype = _VTK_DTYPES[vtk_type]
    except KeyError as exc:
        raise AuditInputError(f"unsupported VTK scalar type: {vtk_type}") from exc
    if dtype.itemsize == 1:
        return dtype
    endian = "<" if byte_order == "LittleEndian" else ">"
    return dtype.newbyteorder(endian)


def _decode_data_array(
    node: ET.Element, *, byte_order: str, header_type: str, compressor: str | None
) -> np.ndarray:
    vtk_type = node.attrib.get("type")
    if vtk_type is None:
        raise AuditInputError("VTK DataArray has no type")
    dtype = _native_dtype(vtk_type, byte_order)
    fmt = node.attrib.get("format", "ascii")
    text = "".join(node.itertext()).strip()
    if fmt == "ascii":
        values = np.fromstring(text, sep=" ", dtype=dtype)
    elif fmt == "binary":
        if compressor:
            raise AuditInputError(
                f"compressed inline VTK data are unsupported ({compressor})"
            )
        raw = base64.b64decode("".join(text.split()), validate=True)
        header_dtype = _native_dtype(header_type, byte_order)
        header_bytes = header_dtype.itemsize
        if len(raw) < header_bytes:
            raise AuditInputError("truncated VTK binary header")
        payload_size = int(np.frombuffer(raw[:header_bytes], dtype=header_dtype)[0])
        payload = raw[header_bytes : header_bytes + payload_size]
        if len(payload) != payload_size:
            # OpenFOAM v2512 writes one known non-standard header for PolyData
            # Int32 connectivity: the header is the byte count multiplied by
            # sizeof(label), while the following payload is complete.  Accept
            # only that exact, self-consistent relation; all other short
            # payloads remain hard failures.
            available = raw[header_bytes:]
            if payload_size == len(available) * dtype.itemsize:
                payload = available
                payload_size = len(payload)
            else:
                raise AuditInputError("truncated VTK binary payload")
        if payload_size % dtype.itemsize:
            raise AuditInputError("VTK payload size is not aligned to its scalar type")
        values = np.frombuffer(payload, dtype=dtype).copy()
    else:
        raise AuditInputError(f"unsupported VTK DataArray format: {fmt}")

    ncomp = int(node.attrib.get("NumberOfComponents", "1"))
    if ncomp < 1 or values.size % ncomp:
        raise AuditInputError("VTK DataArray component count is inconsistent")
    if ncomp > 1:
        values = values.reshape((-1, ncomp))
    return values


def _named_arrays(
    parent: ET.Element | None,
    *,
    byte_order: str,
    header_type: str,
    compressor: str | None,
) -> dict[str, np.ndarray]:
    if parent is None:
        return {}
    result: dict[str, np.ndarray] = {}
    for node in parent.findall("DataArray"):
        name = node.attrib.get("Name")
        if name:
            result[name] = _decode_data_array(
                node,
                byte_order=byte_order,
                header_type=header_type,
                compressor=compressor,
            )
    return result


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    area_vector = np.zeros(3, dtype=float)
    for first, second in zip(points, np.roll(points, -1, axis=0)):
        area_vector += np.cross(first, second)
    return 0.5 * float(np.linalg.norm(area_vector))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _time_from_parent(path: Path) -> tuple[float, str]:
    name = path.parent.name
    try:
        return float(name), name
    except ValueError as exc:
        raise AuditInputError(
            f"surface file parent is not an OpenFOAM time directory: {path}"
        ) from exc


def _merge_largest_component(intervals: list[tuple[float, float, float]]) -> float:
    """Return largest x-contiguous resolved area.

    Each tuple is ``(xmin, xmax, area)``.  Campaign-2 sampled faces span the
    one-cell extrusion, so x-contiguity is the relevant 2-D support test.
    """

    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    scale = max(1.0, max(abs(v) for item in ordered for v in item[:2]))
    tolerance = 128.0 * np.finfo(float).eps * scale
    _, current_right, current_area = ordered[0]
    largest = current_area
    for left, right, area in ordered[1:]:
        if left <= current_right + tolerance:
            current_right = max(current_right, right)
            current_area += area
        else:
            largest = max(largest, current_area)
            current_right = right
            current_area = area
        largest = max(largest, current_area)
    return float(largest)


def read_surface_frame(
    path: Path,
    *,
    mouth_x_min_m: float,
    mouth_x_max_m: float,
) -> SurfaceFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.attrib.get("type") != "PolyData":
        raise AuditInputError(f"expected VTK PolyData, got {root.attrib.get('type')}")
    byte_order = root.attrib.get("byte_order", "LittleEndian")
    header_type = root.attrib.get("header_type", "UInt32")
    compressor = root.attrib.get("compressor")
    pieces = root.findall("./PolyData/Piece")
    if len(pieces) != 1:
        raise AuditInputError(f"expected one PolyData Piece, found {len(pieces)}")
    piece = pieces[0]

    point_node = piece.find("./Points/DataArray")
    if point_node is None:
        raise AuditInputError("PolyData has no point coordinates")
    points = _decode_data_array(
        point_node,
        byte_order=byte_order,
        header_type=header_type,
        compressor=compressor,
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise AuditInputError("point coordinates are not three-component vectors")

    polys = _named_arrays(
        piece.find("Polys"),
        byte_order=byte_order,
        header_type=header_type,
        compressor=compressor,
    )
    if "connectivity" not in polys or "offsets" not in polys:
        raise AuditInputError("PolyData polygons lack connectivity or offsets")
    connectivity = np.asarray(polys["connectivity"], dtype=np.int64).reshape(-1)
    offsets = np.asarray(polys["offsets"], dtype=np.int64).reshape(-1)
    if len(offsets) != int(piece.attrib.get("NumberOfPolys", len(offsets))):
        raise AuditInputError("polygon offset count does not match NumberOfPolys")
    starts = np.concatenate(([0], offsets[:-1]))

    cell_data = _named_arrays(
        piece.find("CellData"),
        byte_order=byte_order,
        header_type=header_type,
        compressor=compressor,
    )
    point_data = _named_arrays(
        piece.find("PointData"),
        byte_order=byte_order,
        header_type=header_type,
        compressor=compressor,
    )
    field_data = _named_arrays(
        root.find("./PolyData/FieldData"),
        byte_order=byte_order,
        header_type=header_type,
        compressor=compressor,
    )

    parent_time, source_time_dir = _time_from_parent(path)
    if "TimeValue" in field_data and np.asarray(field_data["TimeValue"]).size:
        time_s = float(np.asarray(field_data["TimeValue"]).reshape(-1)[0])
        if not math.isclose(time_s, parent_time, abs_tol=TIME_MATCH_ATOL_S):
            raise AuditInputError(
                f"VTK time {time_s} disagrees with source directory {source_time_dir}"
            )
    else:
        time_s = parent_time

    alpha_source = cell_data.get("alpha.water")
    alpha_is_cell = alpha_source is not None
    if alpha_source is None:
        alpha_source = point_data.get("alpha.water")
    velocity_source = cell_data.get("U")
    velocity_is_cell = velocity_source is not None
    if velocity_source is None:
        velocity_source = point_data.get("U")
    if alpha_is_cell and velocity_is_cell:
        data_basis = "cell"
    elif not alpha_is_cell and not velocity_is_cell:
        data_basis = "point-averaged-to-polygon"
    else:
        data_basis = "mixed-cell-and-point"

    missing: list[str] = []
    if alpha_source is None:
        missing.append("alpha.water")
    if velocity_source is None:
        missing.append("U")

    weighted_elements: list[dict[str, float]] = []
    z_values: list[float] = []
    for polygon_index, (start, stop) in enumerate(zip(starts, offsets)):
        ids = connectivity[int(start) : int(stop)]
        polygon_points = points[ids]
        z_values.extend(float(value) for value in polygon_points[:, 2])
        area = _polygon_area(polygon_points)
        x_min = float(np.min(polygon_points[:, 0]))
        x_max = float(np.max(polygon_points[:, 0]))
        x_span = x_max - x_min
        overlap = max(0.0, min(x_max, mouth_x_max_m) - max(x_min, mouth_x_min_m))
        if area <= 0.0 or overlap <= 0.0 or x_span <= 0.0:
            continue
        opening_area = area * overlap / x_span
        element: dict[str, float] = {
            "area": opening_area,
            "x_min": max(x_min, mouth_x_min_m),
            "x_max": min(x_max, mouth_x_max_m),
        }
        if not missing:
            if alpha_is_cell:
                alpha = float(np.asarray(alpha_source)[polygon_index])
            else:
                alpha = float(np.mean(np.asarray(alpha_source)[ids]))
            if velocity_is_cell:
                velocity = np.asarray(velocity_source)[polygon_index]
            else:
                velocity = np.mean(np.asarray(velocity_source)[ids], axis=0)
            element["alpha"] = alpha
            element["uz"] = float(velocity[2])
        weighted_elements.append(element)

    opening_area = float(sum(item["area"] for item in weighted_elements))
    element_areas = [item["area"] for item in weighted_elements]
    min_element_area = min(element_areas) if element_areas else None
    plane_z_min = min(z_values) if z_values else math.nan
    plane_z_max = max(z_values) if z_values else math.nan

    if missing:
        max_alpha = None
        max_alpha_uz = None
        q_positive = None
        resolved_area = None
        largest_component = None
        below_zero = None
        above_one = None
    else:
        alphas = np.array([item["alpha"] for item in weighted_elements], dtype=float)
        uz = np.array([item["uz"] for item in weighted_elements], dtype=float)
        areas = np.array([item["area"] for item in weighted_elements], dtype=float)
        alpha_physical = np.clip(alphas, 0.0, 1.0)
        positive_alpha_uz = alpha_physical * np.maximum(uz, 0.0)
        max_alpha = float(np.max(alphas)) if len(alphas) else None
        max_alpha_uz = float(np.max(positive_alpha_uz)) if len(alphas) else None
        q_positive = float(np.sum(positive_alpha_uz * areas))
        resolved = (alphas >= ALPHA_INTERFACE) & (uz > 0.0)
        resolved_area = float(np.sum(areas[resolved]))
        intervals = [
            (item["x_min"], item["x_max"], item["area"])
            for item, is_resolved in zip(weighted_elements, resolved)
            if is_resolved
        ]
        largest_component = _merge_largest_component(intervals)
        below_zero = int(np.count_nonzero(alphas < 0.0))
        above_one = int(np.count_nonzero(alphas > 1.0))

    return SurfaceFrame(
        time_s=time_s,
        source_time_directory=source_time_dir,
        path=path,
        sha256=_sha256(path),
        plane_z_min_m=plane_z_min,
        plane_z_max_m=plane_z_max,
        opening_area_m2=opening_area,
        n_opening_elements=len(weighted_elements),
        min_opening_element_area_m2=min_element_area,
        max_alpha=max_alpha,
        max_positive_alpha_uz_m_s=max_alpha_uz,
        positive_alpha_weighted_flow_m3_s=q_positive,
        resolved_upward_area_m2=resolved_area,
        largest_resolved_component_area_m2=largest_component,
        alpha_below_zero_count=below_zero,
        alpha_above_one_count=above_one,
        data_basis=data_basis,
        missing=tuple(missing),
    )


def _numeric_time_directories(source_case: Path) -> tuple[str, list[tuple[float, str]]]:
    processor0 = source_case / "processor0"
    root = processor0 if processor0.is_dir() else source_case
    layout = "parallel_decomposed" if processor0.is_dir() else "serial_reconstructed"
    values: list[tuple[float, str]] = []
    for child in root.iterdir():
        if child.is_dir() and _NUMERIC_DIR.match(child.name):
            values.append((float(child.name), child.name))
    values.sort()
    return layout, values


def _parse_solver_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "available": False,
            "last_time_s": None,
            "normal_end": None,
            "fatal_error": None,
            "fatal_matches": [],
        }
    last_time: float | None = None
    normal_end = False
    fatal_matches: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            match = _TIME_LINE.match(stripped)
            if match:
                last_time = float(match.group(1))
            if stripped == "End":
                normal_end = True
            # OpenFOAM always prints "trapFpe: ... enabled" during startup;
            # this is explicitly not an error.
            if "trapFpe" not in stripped and _TRUE_FATAL.search(stripped):
                if len(fatal_matches) < 20:
                    fatal_matches.append(stripped[:500])
    return {
        "path": str(path),
        "available": True,
        "last_time_s": last_time,
        "normal_end": normal_end,
        "fatal_error": bool(fatal_matches),
        "fatal_matches": fatal_matches,
    }


def _same_time_set(
    expected: Sequence[tuple[float, str]], sampled: Sequence[SurfaceFrame]
) -> tuple[list[str], list[str]]:
    remaining = list(sampled)
    missing: list[str] = []
    for expected_time, expected_name in expected:
        match_index = next(
            (
                index
                for index, frame in enumerate(remaining)
                if math.isclose(frame.time_s, expected_time, abs_tol=TIME_MATCH_ATOL_S)
            ),
            None,
        )
        if match_index is None:
            missing.append(expected_name)
        else:
            remaining.pop(match_index)
    extra = [frame.source_time_directory for frame in remaining]
    return missing, extra


def _first_time_at_or_above(
    times: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    return float(times[indices[0]]) if len(indices) else None


def audit(
    *,
    case_config_path: Path,
    surface_root: Path,
    source_case: Path,
    solver_log: Path | None,
) -> dict[str, Any]:
    config = json.loads(case_config_path.read_text(encoding="utf-8"))
    geometry = config["physical_geometry_m"]
    mapping = config["planar_mapping"]
    mesh = config["mesh_m"]
    declared_end_time_raw = config.get("simulation", {}).get("end_time_s")
    declared_end_time = (
        None
        if declared_end_time_raw is None
        else float(declared_end_time_raw)
    )
    if declared_end_time is not None and (
        not math.isfinite(declared_end_time) or declared_end_time <= 0.0
    ):
        raise AuditInputError("simulation.end_time_s must be finite and positive")
    tee_x = float(geometry["tee_axis_x"])
    physical_rim_z = float(geometry["riser_rim_z"])
    area_width = float(mapping["area_equivalent_riser_width_m"])
    extrusion = float(mapping["extrusion_m"])
    physical_diameter = float(mapping["physical_riser_diameter_m"])
    physical_opening_area = math.pi * physical_diameter**2 / 4.0
    riser_dx = float(mesh["riser_dx"])
    external_dz = float(mesh["external_dz"])
    mouth_min = tee_x - 0.5 * area_width
    mouth_max = tee_x + 0.5 * area_width

    files = sorted(
        surface_root.glob("*/physicalRim.vtp"),
        key=lambda item: float(item.parent.name),
    )
    if not files:
        raise AuditInputError(f"no physicalRim.vtp frames below {surface_root}")
    frames = [
        read_surface_frame(
            path,
            mouth_x_min_m=mouth_min,
            mouth_x_max_m=mouth_max,
        )
        for path in files
    ]
    frames.sort(key=lambda frame: frame.time_s)
    if any(a.time_s >= b.time_s for a, b in zip(frames, frames[1:])):
        raise AuditInputError("surface times must be unique and strictly increasing")

    source_layout, source_times = _numeric_time_directories(source_case)
    missing_source_times, extra_surface_times = _same_time_set(source_times, frames)
    log_status = _parse_solver_log(solver_log)

    expected_opening_area = area_width * extrusion
    physical_equivalent_scale = physical_opening_area / expected_opening_area
    area_errors = [
        abs(frame.opening_area_m2 - expected_opening_area) / expected_opening_area
        for frame in frames
    ]
    plane_errors = [
        max(
            abs(frame.plane_z_min_m - (physical_rim_z + RIM_SAMPLE_OFFSET_M)),
            abs(frame.plane_z_max_m - (physical_rim_z + RIM_SAMPLE_OFFSET_M)),
        )
        for frame in frames
    ]
    missing_fields = sorted({name for frame in frames for name in frame.missing})
    geometry_valid = max(area_errors) <= 5.0e-4 and max(plane_errors) <= 2.0e-6
    complete_time_coverage = not missing_source_times and not extra_surface_times

    minimum_face_area = riser_dx * extrusion
    adjacent_cell_volume = minimum_face_area * external_dz
    support_gate_area = AREA_SUPPORT_FRACTION * minimum_face_area

    times = np.array([frame.time_s for frame in frames], dtype=float)
    if not missing_fields:
        flows = np.array(
            [frame.positive_alpha_weighted_flow_m3_s for frame in frames], dtype=float
        )
        cumulative = np.zeros_like(times)
        if len(times) > 1:
            cumulative[1:] = np.cumsum(
                0.5 * (flows[1:] + flows[:-1]) * np.diff(times)
            )
        interface_events = np.array(
            [
                bool(
                    frame.largest_resolved_component_area_m2 is not None
                    and frame.largest_resolved_component_area_m2 >= support_gate_area
                    and frame.positive_alpha_weighted_flow_m3_s is not None
                    and frame.positive_alpha_weighted_flow_m3_s > 0.0
                )
                for frame in frames
            ]
        )
        first_interface_index = (
            int(np.flatnonzero(interface_events)[0]) if np.any(interface_events) else None
        )
        first_interface_time = (
            float(times[first_interface_index]) if first_interface_index is not None else None
        )
        first_volume_time = _first_time_at_or_above(
            times, cumulative, adjacent_cell_volume
        )
        gate_time: float | None = None
        if first_interface_index is not None:
            qualifying = np.flatnonzero(
                (np.arange(len(times)) >= first_interface_index)
                & (cumulative >= adjacent_cell_volume)
            )
            if len(qualifying):
                gate_time = float(times[qualifying[0]])
        max_alpha = max(
            frame.max_alpha for frame in frames if frame.max_alpha is not None
        )
        max_alpha_frame = max(
            frames,
            key=lambda frame: -math.inf if frame.max_alpha is None else frame.max_alpha,
        )
        max_flow_frame = max(
            frames,
            key=lambda frame: (
                -math.inf
                if frame.positive_alpha_weighted_flow_m3_s is None
                else frame.positive_alpha_weighted_flow_m3_s
            ),
        )
        total_positive_volume = float(cumulative[-1])
        resolved_crossing = gate_time is not None
    else:
        flows = np.full(len(times), np.nan)
        cumulative = np.full(len(times), np.nan)
        first_interface_time = None
        first_volume_time = None
        gate_time = None
        max_alpha = None
        max_alpha_frame = None
        max_flow_frame = None
        total_positive_volume = None
        resolved_crossing = None

    normal_end = log_status["normal_end"] is True
    fatal_free = log_status["fatal_error"] is False
    base_evidence_complete = (
        normal_end
        and fatal_free
        and complete_time_coverage
        and geometry_valid
        and not missing_fields
    )
    last_computed_time = max(
        float(times[-1]),
        float(log_status["last_time_s"])
        if log_status["last_time_s"] is not None
        else -math.inf,
    )
    declared_end_reached = bool(
        declared_end_time is not None
        and last_computed_time
        >= declared_end_time
        - max(TIME_MATCH_ATOL_S, 1.0e-12 * declared_end_time)
    )
    # A resolved ejection is irreversible classification evidence, so a
    # normally ended run may establish GEYSER before its planned tail.  The
    # absence of ejection is different: NO_GEYSER is final only after the
    # declared observation window has actually been completed.
    positive_evidence_complete = bool(
        base_evidence_complete and resolved_crossing is True
    )
    negative_evidence_complete = bool(
        base_evidence_complete
        and resolved_crossing is False
        and declared_end_reached
    )
    evidence_complete = positive_evidence_complete or negative_evidence_complete
    if not geometry_valid or missing_fields or not complete_time_coverage:
        classification = "INDETERMINATE_EVIDENCE_GAP"
    elif evidence_complete:
        classification = "GEYSER" if resolved_crossing else "NO_GEYSER"
    elif resolved_crossing:
        classification = "PROVISIONAL_RESOLVED_CROSSING_RUN_INCOMPLETE"
    else:
        classification = "INCOMPLETE_NO_FINAL_CLASSIFICATION"

    frame_records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        frame_records.append(
            {
                "time_s": frame.time_s,
                "source_time_directory": frame.source_time_directory,
                "surface_file": str(frame.path),
                "surface_sha256": frame.sha256,
                "plane_z_min_m": frame.plane_z_min_m,
                "plane_z_max_m": frame.plane_z_max_m,
                "opening_area_m2": frame.opening_area_m2,
                "n_opening_elements": frame.n_opening_elements,
                "min_opening_element_area_m2": frame.min_opening_element_area_m2,
                "max_alpha": frame.max_alpha,
                "max_positive_alpha_uz_m_s": frame.max_positive_alpha_uz_m_s,
                "positive_alpha_weighted_flow_m3_s": (
                    frame.positive_alpha_weighted_flow_m3_s
                ),
                "cumulative_positive_liquid_volume_m3": (
                    None if np.isnan(cumulative[index]) else float(cumulative[index])
                ),
                "resolved_upward_area_m2": frame.resolved_upward_area_m2,
                "largest_resolved_component_area_m2": (
                    frame.largest_resolved_component_area_m2
                ),
                "alpha_below_zero_count": frame.alpha_below_zero_count,
                "alpha_above_one_count": frame.alpha_above_one_count,
                "data_basis": frame.data_basis,
                "missing": list(frame.missing),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": config.get("case_id"),
        "paper_run": config.get("paper_run"),
        "decision": {
            "classification": classification,
            "final": evidence_complete,
            "declared_observation_end_reached": declared_end_reached,
            "resolved_crossing_gate_pass": resolved_crossing,
            "first_interface_supported_upward_crossing_time_s": first_interface_time,
            "first_one_cell_volume_time_s": first_volume_time,
            "first_full_gate_time_s": gate_time,
            "experimental_label_used": False,
            "superseded_98_percent_level_used": False,
        },
        "metrics": {
            "maximum_rim_plane_alpha": max_alpha,
            "maximum_rim_plane_alpha_time_s": (
                None if max_alpha_frame is None else max_alpha_frame.time_s
            ),
            "maximum_positive_alpha_weighted_flow_m3_s": (
                None
                if max_flow_frame is None
                else max_flow_frame.positive_alpha_weighted_flow_m3_s
            ),
            "maximum_positive_alpha_weighted_flow_time_s": (
                None if max_flow_frame is None else max_flow_frame.time_s
            ),
            "cumulative_positive_liquid_volume_m3": total_positive_volume,
            "positive_flow_per_unit_depth_m2_s": (
                None
                if max_flow_frame is None
                else max_flow_frame.positive_alpha_weighted_flow_m3_s / extrusion
            ),
            "cumulative_positive_volume_per_unit_depth_m2": (
                None
                if total_positive_volume is None
                else total_positive_volume / extrusion
            ),
            "physical_circular_equivalent_peak_flow_m3_s": (
                None
                if max_flow_frame is None
                else max_flow_frame.positive_alpha_weighted_flow_m3_s
                * physical_equivalent_scale
            ),
            "physical_circular_equivalent_cumulative_volume_m3": (
                None
                if total_positive_volume is None
                else total_positive_volume * physical_equivalent_scale
            ),
            "opening_area_m2": expected_opening_area,
            "minimum_resolved_face_area_m2": minimum_face_area,
            "one_adjacent_cell_volume_m3": adjacent_cell_volume,
        },
        "uniform_gate": {
            "description": (
                "A resolved physical-rim crossing requires an upward alpha>=0.5 "
                "component covering at least 95% of one rim face and cumulative "
                "positive integral(alpha*Uz dA dt) of at least one adjacent "
                "normal-direction finite-volume cell."
            ),
            "alpha_interface": ALPHA_INTERFACE,
            "minimum_component_area_fraction_of_one_face": AREA_SUPPORT_FRACTION,
            "minimum_component_area_m2": support_gate_area,
            "minimum_cumulative_volume_m3": adjacent_cell_volume,
            "positive_flow_definition": "integral_opening(alpha*max(Uz,0) dA)",
            "cumulative_definition": "trapezoidal integral of stored-field Q+",
            "stored_field_basis": (
                "cell-centred alpha.water and U sampled on the rim plane; this "
                "is an alpha-weighted advective-flow audit, not a recovered "
                "OpenFOAM face alphaPhi ledger"
            ),
            "same_formula_for_all_cases": True,
        },
        "geometry_and_mapping": {
            "physical_rim_z_m": physical_rim_z,
            "sample_plane_z_m": physical_rim_z + RIM_SAMPLE_OFFSET_M,
            "sample_offset_above_rim_m": RIM_SAMPLE_OFFSET_M,
            "tee_axis_x_m": tee_x,
            "mouth_x_interval_m": [mouth_min, mouth_max],
            "physical_riser_diameter_m": physical_diameter,
            "physical_circular_opening_area_m2": physical_opening_area,
            "area_equivalent_riser_width_m": area_width,
            "area_mapping_formula": mapping["formula"],
            "extrusion_m": extrusion,
            "riser_dx_m": riser_dx,
            "external_dz_m": external_dz,
            "expected_opening_area_m2": expected_opening_area,
            "model_to_physical_circular_area_scale": physical_equivalent_scale,
            "maximum_relative_sampled_area_error": max(area_errors),
            "maximum_plane_coordinate_error_m": max(plane_errors),
            "geometry_gate_pass": geometry_valid,
        },
        "coverage": {
            "source_case": str(source_case),
            "source_layout": source_layout,
            "surface_root": str(surface_root),
            "source_time_directory_count": len(source_times),
            "sampled_time_count": len(frames),
            "first_sample_time_s": float(times[0]),
            "last_sample_time_s": float(times[-1]),
            "declared_observation_end_time_s": declared_end_time,
            "last_computed_time_s": last_computed_time,
            "declared_observation_end_reached": declared_end_reached,
            "missing_source_time_directories": missing_source_times,
            "extra_surface_time_directories": extra_surface_times,
            "complete_stored_time_coverage": complete_time_coverage,
            "solver_log": log_status,
        },
        "missing_metrics": missing_fields,
        "frames": frame_records,
    }


def _fmt(value: Any, digits: int = 10) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _event_time_text(value: float | None) -> str:
    return "not observed" if value is None else f"`{_fmt(value)} s`"


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    metrics = report["metrics"]
    geometry = report["geometry_and_mapping"]
    coverage = report["coverage"]
    gate = report["uniform_gate"]
    missing = report["missing_metrics"]
    lines = [
        f"# {report.get('paper_run') or report.get('case_id')} physical-rim outlet audit",
        "",
        "## Decision",
        "",
        f"- Classification: **{decision['classification']}**",
        f"- Final evidence gate: {_fmt(decision['final'])}",
        "- Scope of final gate: physical outlet classification only",
        f"- Resolved crossing gate: {_fmt(decision['resolved_crossing_gate_pass'])}",
        "- Experimental outcome used in the decision: no",
        "- 98%-of-rim liquid-level criterion used: no",
        "",
        "## Physical outlet metrics",
        "",
        f"- Maximum `alpha.water` on the physical opening: `{_fmt(metrics['maximum_rim_plane_alpha'])}`",
        f"- Peak positive alpha-weighted flow: `{_fmt(metrics['maximum_positive_alpha_weighted_flow_m3_s'])} m3/s`",
        f"- Cumulative positive liquid volume: `{_fmt(metrics['cumulative_positive_liquid_volume_m3'])} m3`",
        f"- Physical-circular equivalent peak flow: `{_fmt(metrics['physical_circular_equivalent_peak_flow_m3_s'])} m3/s`",
        f"- Physical-circular equivalent cumulative volume: `{_fmt(metrics['physical_circular_equivalent_cumulative_volume_m3'])} m3`",
        f"- First interface-supported upward crossing: {_event_time_text(decision['first_interface_supported_upward_crossing_time_s'])}",
        f"- First one-cell-volume passage: {_event_time_text(decision['first_one_cell_volume_time_s'])}",
        f"- First full-gate time: {_event_time_text(decision['first_full_gate_time_s'])}",
        "",
        "## Uniform numerical-resolution gate",
        "",
        gate["description"],
        "",
        f"- Interface value: `alpha.water >= {gate['alpha_interface']}`",
        f"- Minimum contiguous upward area: `{_fmt(gate['minimum_component_area_m2'])} m2`",
        f"- Minimum cumulative passage: `{_fmt(gate['minimum_cumulative_volume_m3'])} m3`",
        f"- Flow definition: `{gate['positive_flow_definition']}`",
        f"- Stored-field basis: {gate['stored_field_basis']}",
        "",
        "## Geometry and provenance",
        "",
        f"- True physical rim: `z = {_fmt(geometry['physical_rim_z_m'])} m`",
        f"- Sample plane: `z = {_fmt(geometry['sample_plane_z_m'])} m` (offset `{_fmt(geometry['sample_offset_above_rim_m'])} m`)",
        f"- Area-equivalent width: `{_fmt(geometry['area_equivalent_riser_width_m'])} m`",
        f"- 2-D extrusion: `{_fmt(geometry['extrusion_m'])} m`",
        f"- Model-to-physical circular area scale: `{_fmt(geometry['model_to_physical_circular_area_scale'])}`",
        f"- Mapping: `{geometry['area_mapping_formula']}`",
        f"- Source case: `{coverage['source_case']}`",
        f"- Source layout: `{coverage['source_layout']}`",
        f"- Stored/source times sampled: `{coverage['sampled_time_count']}/{coverage['source_time_directory_count']}`",
        f"- Sample interval: `{_fmt(coverage['first_sample_time_s'])}` to `{_fmt(coverage['last_sample_time_s'])} s`",
        f"- Declared observation end: `{_fmt(coverage['declared_observation_end_time_s'])} s`",
        f"- Declared observation end reached: `{_fmt(coverage['declared_observation_end_reached'])}`",
        f"- Normal solver End: `{_fmt(coverage['solver_log']['normal_end'])}`",
        f"- True fatal/NaN evidence: `{_fmt(coverage['solver_log']['fatal_error'])}`",
        "",
        "Each per-frame record, source time-directory name, sampled-surface SHA-256,",
        "and cumulative flux ledger is stored in the companion JSON file.",
    ]
    if missing:
        lines.extend(
            [
                "",
                "## Missing evidence",
                "",
                "The following requested fields/metrics were unavailable and were not inferred from images:",
                "",
                *[f"- `{item}`" for item in missing],
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--surface-root", type=Path, required=True)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit(
        case_config_path=args.case_config,
        surface_root=args.surface_root,
        source_case=args.source_case,
        solver_log=args.solver_log,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
