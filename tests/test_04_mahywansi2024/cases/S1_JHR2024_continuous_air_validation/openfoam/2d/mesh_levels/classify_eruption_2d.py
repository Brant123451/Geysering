#!/usr/bin/env python3
"""Classify the Mahyawansi JHR2024 2-D water-eruption event.

The classifier is intentionally independent of mesh cell count.  It reads
cell-centred ``alpha.water`` (and optionally ``U``) from ASCII ``internal.vtu``
files produced by ``foamToVTK -ascii -no-point-data``.  An optional ``--export``
mode invokes foamToVTK into a temporary directory; it never needs to leave a
VTK tree in the OpenFOAM case.

The hard event definition is frozen in physical units below.  It uses
face-connected wet cells, a mouth-attachment test, a fixed water-equivalent
volume, a 99 % cumulative-water-volume bulk-top height, and a fixed 0.10 s
persistence test.  A single wet cell or a momentarily wet rim cannot pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
CLASSIFIER_VERSION = "2026-08-10.physical-gate-v1"

# Published/source-aligned geometry.
D_M = 0.0254
RIM_Z_M = 1.02
EXTERNAL_X_MIN_M = -0.10
EXTERNAL_X_MAX_M = 0.10
EXTRUSION_THICKNESS_M = math.pi * D_M / 4.0
PIPE_AREA_M2 = math.pi * D_M**2 / 4.0

# Declared, mesh-independent event-classification thresholds.  These are not
# claimed as published experimental tolerances.
ALPHA_WET = 0.5
LAUNCH_BAND_TOP_M = RIM_Z_M + D_M / 4.0
MIN_BULK_TOP_Z_M = RIM_Z_M + D_M / 2.0
MIN_COMPONENT_WATER_VOLUME_M3 = PIPE_AREA_M2 * D_M / 4.0
BULK_TOP_WATER_QUANTILE = 0.99
PERSISTENCE_S = 0.10
COMMON_SAMPLE_INTERVAL_S = 0.10
NO_FINAL_NO_EVENT_BEFORE_S = 12.0
PLANNED_STAGE2_DURATION_S = 25.0

MARKER_NAMES = (
    "ERUPTION_ACCEPTED",
    "ERUPTION_FAILED_NO_EVENT",
    "ERUPTION_INCONCLUSIVE",
)


@dataclass(frozen=True)
class FrameRef:
    time_s: float
    path: Path


@dataclass
class Geometry:
    cell_vertices: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray
    xmid: np.ndarray
    zmid: np.ndarray
    volumes: np.ndarray
    external: np.ndarray
    launch_band: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]


@dataclass
class FrameFields:
    alpha: np.ndarray
    velocity: np.ndarray | None


def _data_array(
    raw: str,
    name: str,
    *,
    scope: str | None = None,
    required: bool = True,
) -> np.ndarray | None:
    text = raw
    if scope is not None:
        block = re.search(
            rf"<{scope}\b[^>]*>(.*?)</{scope}>", text, flags=re.S | re.I
        )
        if not block:
            if required:
                raise RuntimeError(f"missing <{scope}> block")
            return None
        text = block.group(1)
    match = re.search(
        rf"<DataArray\b(?P<a>[^>]*)\bName=['\"]{re.escape(name)}['\"]"
        rf"(?P<b>[^>]*)>(?P<body>.*?)</DataArray>",
        text,
        flags=re.S | re.I,
    )
    if not match:
        if required:
            raise RuntimeError(f"missing DataArray {name!r}")
        return None
    attributes = match.group("a") + match.group("b")
    if not re.search(r"\bformat=['\"]ascii['\"]", attributes, flags=re.I):
        raise RuntimeError(
            f"DataArray {name!r} is not ASCII; export with foamToVTK -ascii"
        )
    return np.fromstring(match.group("body"), sep=" ", dtype=np.float64)


def _hex_faces(vertices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if len(vertices) != 8:
        raise RuntimeError(f"expected 8-node VTK_HEXAHEDRON, got {len(vertices)}")
    v = tuple(int(item) for item in vertices)
    return (
        (v[0], v[1], v[2], v[3]),
        (v[4], v[5], v[6], v[7]),
        (v[0], v[1], v[5], v[4]),
        (v[1], v[2], v[6], v[5]),
        (v[2], v[3], v[7], v[6]),
        (v[3], v[0], v[4], v[7]),
    )


def _build_face_adjacency(
    cell_vertices: np.ndarray, external: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    neighbours: list[list[int]] = [[] for _ in range(len(cell_vertices))]
    face_owner: dict[tuple[int, ...], int] = {}
    for cell in np.flatnonzero(external):
        for face in _hex_faces(cell_vertices[cell]):
            key = tuple(sorted(face))
            other = face_owner.pop(key, None)
            if other is None:
                face_owner[key] = int(cell)
            else:
                neighbours[cell].append(other)
                neighbours[other].append(int(cell))
    return tuple(tuple(items) for items in neighbours)


def read_geometry_and_fields(path: Path) -> tuple[Geometry, FrameFields]:
    raw = path.read_text(encoding="utf-8")
    points_raw = _data_array(raw, "Points")
    connectivity_raw = _data_array(raw, "connectivity")
    offsets_raw = _data_array(raw, "offsets")
    types_raw = _data_array(raw, "types")
    assert points_raw is not None
    assert connectivity_raw is not None
    assert offsets_raw is not None
    assert types_raw is not None

    points = points_raw.reshape(-1, 3)
    connectivity = connectivity_raw.astype(np.int64)
    offsets = offsets_raw.astype(np.int64)
    cell_types = types_raw.astype(np.int64)
    counts = np.diff(np.r_[0, offsets])
    if not np.all(counts == 8) or not np.all(cell_types == 12):
        unique_counts = sorted(set(int(value) for value in counts))
        unique_types = sorted(set(int(value) for value in cell_types))
        raise RuntimeError(
            "classifier currently requires VTK_HEXAHEDRON cells; "
            f"counts={unique_counts}, types={unique_types}"
        )
    cells = connectivity.reshape(len(offsets), 8)
    vertices = points[cells]
    xmin = vertices[:, :, 0].min(axis=1)
    xmax = vertices[:, :, 0].max(axis=1)
    ymin = vertices[:, :, 1].min(axis=1)
    ymax = vertices[:, :, 1].max(axis=1)
    zmin = vertices[:, :, 2].min(axis=1)
    zmax = vertices[:, :, 2].max(axis=1)
    xmid = 0.5 * (xmin + xmax)
    zmid = 0.5 * (zmin + zmax)
    volumes = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise RuntimeError(f"{path}: non-positive or non-finite cell volume")

    tolerance = 1.0e-9
    external = (
        (xmid >= EXTERNAL_X_MIN_M - tolerance)
        & (xmid <= EXTERNAL_X_MAX_M + tolerance)
        & (zmin >= RIM_Z_M - tolerance)
    )
    if not np.any(external):
        raise RuntimeError(f"{path}: no cells found in the configured external domain")
    launch_band = (
        external
        & (np.abs(xmid) <= D_M / 2.0 + tolerance)
        & (zmid <= LAUNCH_BAND_TOP_M + tolerance)
    )
    if not np.any(launch_band):
        raise RuntimeError(f"{path}: no cells found in the fixed launch band")

    fields = read_fields_from_raw(raw, len(offsets), path)
    geometry = Geometry(
        cell_vertices=cells,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        xmid=xmid,
        zmid=zmid,
        volumes=volumes,
        external=external,
        launch_band=launch_band,
        adjacency=_build_face_adjacency(cells, external),
    )
    return geometry, fields


def read_fields_from_raw(raw: str, cell_count: int, path: Path) -> FrameFields:
    alpha_raw = _data_array(raw, "alpha.water", scope="CellData")
    assert alpha_raw is not None
    if len(alpha_raw) != cell_count:
        raise RuntimeError(
            f"{path}: {len(alpha_raw)} alpha values for {cell_count} cells"
        )
    velocity_raw = _data_array(
        raw, "U", scope="CellData", required=False
    )
    velocity: np.ndarray | None
    if velocity_raw is None:
        velocity = None
    else:
        if len(velocity_raw) != 3 * cell_count:
            raise RuntimeError(
                f"{path}: {len(velocity_raw)} U values for {cell_count} cells"
            )
        velocity = velocity_raw.reshape(cell_count, 3)
    return FrameFields(alpha=alpha_raw, velocity=velocity)


def read_fields(path: Path, cell_count: int) -> FrameFields:
    return read_fields_from_raw(path.read_text(encoding="utf-8"), cell_count, path)


def _internal_vtu_for_vtm(vtm_path: Path) -> Path:
    raw = vtm_path.read_text(encoding="utf-8")
    match = re.search(
        r"<DataSet\b[^>]*\bname=['\"]internal['\"][^>]*\bfile=['\"]([^'\"]+)['\"]",
        raw,
        flags=re.I,
    )
    if not match:
        match = re.search(
            r"<DataSet\b[^>]*\bfile=['\"]([^'\"]*internal\.vtu)['\"]",
            raw,
            flags=re.I,
        )
    if not match:
        raise RuntimeError(f"{vtm_path}: no internal.vtu reference")
    result = (vtm_path.parent / match.group(1)).resolve()
    if not result.is_file():
        raise RuntimeError(f"{vtm_path}: referenced internal VTU does not exist: {result}")
    return result


def find_vtk_frames(vtk_dir: Path) -> tuple[Path, list[FrameRef]]:
    series_files = sorted(vtk_dir.rglob("*.vtm.series"))
    if len(series_files) != 1:
        raise RuntimeError(
            f"expected exactly one *.vtm.series below {vtk_dir}, "
            f"found {len(series_files)}"
        )
    series_path = series_files[0]
    payload = json.loads(series_path.read_text(encoding="utf-8"))
    frames: list[FrameRef] = []
    for entry in payload.get("files", []):
        vtm = (series_path.parent / str(entry["name"])).resolve()
        if not vtm.is_file():
            raise RuntimeError(f"series entry is missing: {vtm}")
        frames.append(FrameRef(float(entry["time"]), _internal_vtu_for_vtm(vtm)))
    if not frames:
        raise RuntimeError(f"{series_path}: no frames")
    frames.sort(key=lambda item: item.time_s)
    return series_path, frames


def _read_stage2_start(case_dir: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    marker = case_dir / "STAGE1_ACCEPTED_TIME"
    if marker.is_file():
        return float(marker.read_text(encoding="utf-8").strip())
    raise RuntimeError(
        "Stage-2 start time is unknown. Pass --stage2-start or provide "
        "STAGE1_ACCEPTED_TIME in the case directory."
    )


def select_common_time_frames(
    frames: Sequence[FrameRef], stage2_start: float
) -> tuple[list[tuple[int, float, FrameRef]], list[float]]:
    available = [frame for frame in frames if frame.time_s >= stage2_start - 1.0e-7]
    if not available:
        raise RuntimeError("no VTK frame at or after Stage-2 start")
    latest = available[-1].time_s
    last_index = int(math.floor((latest - stage2_start) / COMMON_SAMPLE_INTERVAL_S + 1.0e-7))
    selected: list[tuple[int, float, FrameRef]] = []
    missing: list[float] = []
    used_paths: set[Path] = set()
    tolerance = 0.51 * COMMON_SAMPLE_INTERVAL_S
    for sample_index in range(last_index + 1):
        target = stage2_start + sample_index * COMMON_SAMPLE_INTERVAL_S
        nearest = min(available, key=lambda frame: abs(frame.time_s - target))
        if abs(nearest.time_s - target) > tolerance or nearest.path in used_paths:
            missing.append(target)
            continue
        selected.append((sample_index, target, nearest))
        used_paths.add(nearest.path)
    if not selected:
        raise RuntimeError("no frames could be mapped to the common 0.10 s grid")
    return selected, missing


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("weighted quantile requires equal non-empty arrays")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weighted quantile requires positive finite weight")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile * total, side="left"))
    index = min(index, len(order) - 1)
    return float(values[order[index]])


def _launch_components(geometry: Geometry, wet: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(len(wet), dtype=bool)
    components: list[np.ndarray] = []
    for seed in np.flatnonzero(wet & geometry.launch_band):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [int(seed)]
        members: list[int] = []
        while stack:
            cell = stack.pop()
            members.append(cell)
            for neighbour in geometry.adjacency[cell]:
                if wet[neighbour] and not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        components.append(np.asarray(members, dtype=np.int64))
    return components


def classify_frame(
    geometry: Geometry,
    fields: FrameFields,
) -> dict[str, object]:
    alpha = fields.alpha
    if np.any(~np.isfinite(alpha)):
        raise RuntimeError("alpha.water contains NaN/Inf")
    wet = geometry.external & (alpha >= ALPHA_WET)
    components = _launch_components(geometry, wet)
    candidates: list[dict[str, object]] = []
    tolerance = 1.0e-12
    for component in components:
        weights = alpha[component] * geometry.volumes[component]
        water_volume = float(weights.sum())
        bulk_top = _weighted_quantile(
            geometry.zmax[component], weights, BULK_TOP_WATER_QUANTILE
        )
        tip_z = float(geometry.zmax[component].max())
        mean_uz: float | None
        if fields.velocity is None:
            mean_uz = None
        else:
            mean_uz = float(np.dot(weights, fields.velocity[component, 2]) / water_volume)
        volume_pass = water_volume >= MIN_COMPONENT_WATER_VOLUME_M3 - tolerance
        height_pass = bulk_top >= MIN_BULK_TOP_Z_M - tolerance
        candidates.append(
            {
                "cell_count": int(len(component)),
                "water_volume_m3": water_volume,
                "volume_over_minimum": water_volume / MIN_COMPONENT_WATER_VOLUME_M3,
                "bulk_top_q99_z_m": bulk_top,
                "bulk_height_above_rim_m": max(bulk_top - RIM_Z_M, 0.0),
                "tip_z_m": tip_z,
                "tip_height_above_rim_m": max(tip_z - RIM_Z_M, 0.0),
                "water_weighted_uz_m_per_s": mean_uz,
                "volume_pass": bool(volume_pass),
                "height_pass": bool(height_pass),
                "active_raw": bool(volume_pass and height_pass),
            }
        )
    if not candidates:
        return {
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "launch_component_count": 0,
            "component_cell_count": 0,
            "component_water_volume_m3": 0.0,
            "volume_over_minimum": 0.0,
            "bulk_top_q99_z_m": None,
            "bulk_height_above_rim_m": 0.0,
            "tip_z_m": None,
            "tip_height_above_rim_m": 0.0,
            "water_weighted_uz_m_per_s": None,
            "volume_pass": False,
            "height_pass": False,
            "active_raw": False,
        }
    # Never hide an event in a smaller component behind a larger non-passing
    # pool.  Passing components sort first, then by water-equivalent volume.
    chosen = max(
        candidates,
        key=lambda item: (bool(item["active_raw"]), float(item["water_volume_m3"])),
    )
    return {
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "launch_component_count": len(candidates),
        "component_cell_count": chosen["cell_count"],
        "component_water_volume_m3": chosen["water_volume_m3"],
        "volume_over_minimum": chosen["volume_over_minimum"],
        "bulk_top_q99_z_m": chosen["bulk_top_q99_z_m"],
        "bulk_height_above_rim_m": chosen["bulk_height_above_rim_m"],
        "tip_z_m": chosen["tip_z_m"],
        "tip_height_above_rim_m": chosen["tip_height_above_rim_m"],
        "water_weighted_uz_m_per_s": chosen["water_weighted_uz_m_per_s"],
        "volume_pass": chosen["volume_pass"],
        "height_pass": chosen["height_pass"],
        "active_raw": chosen["active_raw"],
    }


def apply_persistence(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        row["active_persistent"] = False
        row["event_id"] = None
    event_id = 0
    start = 0
    while start < len(rows):
        if not bool(rows[start]["active_raw"]):
            start += 1
            continue
        end = start
        while (
            end + 1 < len(rows)
            and bool(rows[end + 1]["active_raw"])
            and int(rows[end + 1]["sample_index"])
            == int(rows[end]["sample_index"]) + 1
        ):
            end += 1
        duration = float(rows[end]["target_time_s"]) - float(rows[start]["target_time_s"])
        if duration >= PERSISTENCE_S - 1.0e-9:
            event_id += 1
            for index in range(start, end + 1):
                rows[index]["active_persistent"] = True
                rows[index]["event_id"] = event_id
        start = end + 1
    return rows


def _finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty eruption time series")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite_or_none(value) for key, value in row.items()})


def _event_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    event_ids = sorted(
        {int(row["event_id"]) for row in rows if row["event_id"] is not None}
    )
    events: list[dict[str, object]] = []
    for event_id in event_ids:
        selected = [row for row in rows if row["event_id"] == event_id]
        peak = max(selected, key=lambda row: float(row["bulk_height_above_rim_m"]))
        events.append(
            {
                "event_id": event_id,
                "start_time_s": float(selected[0]["actual_time_s"]),
                "end_time_s": float(selected[-1]["actual_time_s"]),
                "sampled_duration_s": float(selected[-1]["target_time_s"])
                - float(selected[0]["target_time_s"]),
                "peak_bulk_height_above_rim_m": float(
                    peak["bulk_height_above_rim_m"]
                ),
                "peak_tip_height_above_rim_m": float(peak["tip_height_above_rim_m"]),
                "peak_time_s": float(peak["actual_time_s"]),
                "maximum_component_water_volume_m3": max(
                    float(row["component_water_volume_m3"]) for row in selected
                ),
            }
        )
    return events


def _status_from_rows(
    rows: list[dict[str, object]], missing_targets: Sequence[float], stage2_start: float
) -> tuple[str, str]:
    if any(bool(row["active_persistent"]) for row in rows):
        return "PASS_ERUPTION", "persistent source-aligned eruption detected"
    elapsed = float(rows[-1]["actual_time_s"]) - stage2_start
    expected_count = int(round(PLANNED_STAGE2_DURATION_S / COMMON_SAMPLE_INTERVAL_S)) + 1
    complete_grid = (
        elapsed >= PLANNED_STAGE2_DURATION_S - 0.5 * COMMON_SAMPLE_INTERVAL_S
        and not missing_targets
        and len(rows) >= expected_count
    )
    if complete_grid:
        return (
            "FAIL_NO_ERUPTION",
            "complete 25 s Stage-2 observation contains no persistent eruption",
        )
    if elapsed < NO_FINAL_NO_EVENT_BEFORE_S:
        return (
            "INCONCLUSIVE",
            "less than 12 s of Stage 2 is insufficient for a no-eruption decision",
        )
    return (
        "INCONCLUSIVE",
        "Stage-2 observation is shorter than 25 s or has common-grid gaps",
    )


def _write_marker(case_dir: Path, status: str, summary_path: Path) -> Path:
    marker_for_status = {
        "PASS_ERUPTION": "ERUPTION_ACCEPTED",
        "FAIL_NO_ERUPTION": "ERUPTION_FAILED_NO_EVENT",
        "INCONCLUSIVE": "ERUPTION_INCONCLUSIVE",
    }[status]
    for name in MARKER_NAMES:
        path = case_dir / name
        if path.exists():
            path.unlink()
    marker = case_dir / marker_for_status
    marker.write_text(
        f"status={status}\nsummary={summary_path}\nclassifier={CLASSIFIER_VERSION}\n",
        encoding="utf-8",
    )
    return marker


def _discover_case_time_names(case_dir: Path, stage2_start: float) -> list[tuple[float, str]]:
    times: list[tuple[float, str]] = []
    for child in case_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value >= stage2_start - 1.0e-7:
            times.append((value, child.name))
    if not times:
        raise RuntimeError("case contains no written time at or after Stage-2 start")
    return sorted(times)


def _select_case_time_names(case_dir: Path, stage2_start: float) -> list[str]:
    times = _discover_case_time_names(case_dir, stage2_start)
    latest = times[-1][0]
    last_index = int(math.floor((latest - stage2_start) / COMMON_SAMPLE_INTERVAL_S + 1.0e-7))
    tolerance = 0.51 * COMMON_SAMPLE_INTERVAL_S
    selected: list[str] = []
    used: set[str] = set()
    for index in range(last_index + 1):
        target = stage2_start + index * COMMON_SAMPLE_INTERVAL_S
        value, name = min(times, key=lambda item: abs(item[0] - target))
        if abs(value - target) <= tolerance and name not in used:
            selected.append(name)
            used.add(name)
    if not selected:
        raise RuntimeError("no case times map to the common 0.10 s grid")
    return selected


def export_ascii_vtk(case_dir: Path, stage2_start: float, output_dir: Path) -> None:
    executable = shutil.which("foamToVTK")
    if executable is None:
        raise RuntimeError(
            "foamToVTK is not on PATH; source OpenFOAM v2512 or pass --vtk-dir"
        )
    time_names = _select_case_time_names(case_dir, stage2_start)
    # Use a temporary symlink-only shadow case.  foamToVTK can then use its
    # standard VTK output location without ever creating CASE/VTK or relying
    # on version-specific output-directory options.
    shadow_case = output_dir / "shadowCase"
    shadow_case.mkdir(parents=True, exist_ok=False)
    for name in ("constant", "system", *time_names):
        source = (case_dir / name).resolve()
        if not source.exists():
            raise RuntimeError(f"required case entry is missing: {source}")
        (shadow_case / name).symlink_to(source, target_is_directory=True)
    command = [
        executable,
        "-case",
        str(shadow_case),
        "-ascii",
        "-no-boundary",
        "-no-point-data",
        "-fields",
        "(alpha.water U)",
        "-time",
        ",".join(time_names),
    ]
    subprocess.run(command, cwd=shadow_case, check=True)


def _threshold_payload() -> dict[str, object]:
    return {
        "status": "declared_mesh_independent_classifier_thresholds",
        "diameter_m": D_M,
        "riser_rim_z_m": RIM_Z_M,
        "external_x_bounds_m": [EXTERNAL_X_MIN_M, EXTERNAL_X_MAX_M],
        "alpha_water_wet_threshold": ALPHA_WET,
        "launch_band": {
            "abs_x_max_m": D_M / 2.0,
            "z_min_m": RIM_Z_M,
            "z_max_m": LAUNCH_BAND_TOP_M,
        },
        "minimum_bulk_top_z_m": MIN_BULK_TOP_Z_M,
        "minimum_bulk_height_above_rim_m": D_M / 2.0,
        "bulk_top_definition": (
            "99th percentile of cell upper-face z, weighted by alpha.water*cellVolume, "
            "within the face-connected launch-attached wet component"
        ),
        "bulk_top_water_quantile": BULK_TOP_WATER_QUANTILE,
        "minimum_component_water_equivalent_volume_m3": MIN_COMPONENT_WATER_VOLUME_M3,
        "minimum_component_water_equivalent_area_m2": D_M**2 / 4.0,
        "minimum_component_volume_definition": "A_pipe*(D/4) = pi*D^3/16",
        "connectivity": "shared_mesh_face",
        "persistence_s": PERSISTENCE_S,
        "common_sample_interval_s": COMMON_SAMPLE_INTERVAL_S,
        "no_final_no_event_before_stage2_s": NO_FINAL_NO_EVENT_BEFORE_S,
        "planned_stage2_duration_s": PLANNED_STAGE2_DURATION_S,
    }


def run_classifier(
    case_dir: Path,
    vtk_dir: Path,
    output_dir: Path,
    stage2_start: float,
    *,
    write_marker: bool,
) -> dict[str, object]:
    series_path, frames = find_vtk_frames(vtk_dir)
    selected, missing = select_common_time_frames(frames, stage2_start)

    geometry: Geometry | None = None
    rows: list[dict[str, object]] = []
    expected_cells: int | None = None
    for sample_index, target_time, frame in selected:
        if geometry is None:
            geometry, fields = read_geometry_and_fields(frame.path)
            expected_cells = len(geometry.volumes)
        else:
            assert expected_cells is not None
            fields = read_fields(frame.path, expected_cells)
        metrics = classify_frame(geometry, fields)
        rows.append(
            {
                "sample_index": sample_index,
                "target_time_s": target_time,
                "actual_time_s": frame.time_s,
                "stage2_elapsed_s": frame.time_s - stage2_start,
                "alpha_min": metrics["alpha_min"],
                "alpha_max": metrics["alpha_max"],
                "launch_component_count": metrics["launch_component_count"],
                "component_cell_count": metrics["component_cell_count"],
                "component_water_volume_m3": metrics["component_water_volume_m3"],
                "volume_over_minimum": metrics["volume_over_minimum"],
                "bulk_top_q99_z_m": metrics["bulk_top_q99_z_m"],
                "bulk_height_above_rim_m": metrics["bulk_height_above_rim_m"],
                "tip_z_m": metrics["tip_z_m"],
                "tip_height_above_rim_m": metrics["tip_height_above_rim_m"],
                "water_weighted_uz_m_per_s": metrics["water_weighted_uz_m_per_s"],
                "volume_pass": metrics["volume_pass"],
                "height_pass": metrics["height_pass"],
                "active_raw": metrics["active_raw"],
            }
        )
    apply_persistence(rows)
    status, reason = _status_from_rows(rows, missing, stage2_start)
    events = _event_summaries(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "eruption_timeseries.csv"
    summary_path = output_dir / "eruption_acceptance.json"
    _write_csv(csv_path, rows)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "case_dir": str(case_dir),
        "source_vtk_series": str(series_path),
        "stage2_start_s": stage2_start,
        "last_sample_time_s": float(rows[-1]["actual_time_s"]),
        "observed_stage2_duration_s": float(rows[-1]["actual_time_s"])
        - stage2_start,
        "common_grid_sample_count": len(rows),
        "missing_common_grid_times_s": missing,
        "thresholds": _threshold_payload(),
        "classification": {
            "status": status,
            "reason": reason,
            "eruption_detected": status == "PASS_ERUPTION",
            "event_count": len(events),
        },
        "events": events,
        "outputs": {"timeseries_csv": str(csv_path)},
        "interpretation": {
            "hard_source_branch_expected": "eruption",
            "stable_solver_without_eruption": "physics_alignment_failure",
            "short_or_gapped_run_without_eruption": "inconclusive",
            "note": (
                "This event gate is declared for the 2-D translation; it is not a "
                "published experimental uncertainty band."
            ),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if write_marker:
        marker = _write_marker(case_dir, status, summary_path)
        summary["outputs"]["marker"] = str(marker)  # type: ignore[index]
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    return summary


def _synthetic_geometry(nx: int, dz: float, nz: int) -> Geometry:
    dx = D_M / nx
    y0 = 0.0
    y1 = EXTRUSION_THICKNESS_M
    cells: list[tuple[float, float, float, float, float, float]] = []
    for iz in range(nz):
        z0 = RIM_Z_M + iz * dz
        z1 = z0 + dz
        for ix in range(nx):
            x0 = -D_M / 2.0 + ix * dx
            x1 = x0 + dx
            cells.append((x0, x1, y0, y1, z0, z1))
    bounds = np.asarray(cells, dtype=np.float64)
    count = len(bounds)
    xmin, xmax, ymin, ymax, zmin, zmax = (bounds[:, i] for i in range(6))
    xmid = 0.5 * (xmin + xmax)
    zmid = 0.5 * (zmin + zmax)
    volumes = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for iz in range(nz):
        for ix in range(nx):
            index = iz * nx + ix
            if ix > 0:
                adjacency[index].append(index - 1)
            if ix + 1 < nx:
                adjacency[index].append(index + 1)
            if iz > 0:
                adjacency[index].append(index - nx)
            if iz + 1 < nz:
                adjacency[index].append(index + nx)
    external = np.ones(count, dtype=bool)
    launch = zmid <= LAUNCH_BAND_TOP_M + 1.0e-9
    return Geometry(
        cell_vertices=np.empty((count, 8), dtype=np.int64),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        xmid=xmid,
        zmid=zmid,
        volumes=volumes,
        external=external,
        launch_band=launch,
        adjacency=tuple(tuple(items) for items in adjacency),
    )


def _self_test_level(nx: int, dz: float) -> None:
    nz = max(16, int(math.ceil(D_M / dz)))
    geometry = _synthetic_geometry(nx, dz, nz)
    count = len(geometry.volumes)
    velocity = np.zeros((count, 3), dtype=np.float64)
    velocity[:, 2] = 1.0

    # Merely filling to D/4 reaches the mouth region but is not an eruption.
    rim_only = (geometry.zmax <= RIM_Z_M + D_M / 4.0 + 1.0e-9).astype(float)
    rim_metrics = classify_frame(geometry, FrameFields(rim_only, velocity))
    assert not rim_metrics["active_raw"]

    # A full-bore connected column to D/2 meets fixed height and volume gates.
    # Include the row intersected by the fixed D/2 plane.  The real coarse
    # external dz is rounded from D/4, so its second upper face is 2.7e-5 m
    # above the analytical plane.
    erupted = (geometry.zmin < MIN_BULK_TOP_Z_M - 1.0e-9).astype(float)
    eruption_metrics = classify_frame(geometry, FrameFields(erupted, velocity))
    assert eruption_metrics["active_raw"], eruption_metrics

    # A single high wet cell cannot pass the fixed physical volume gate.
    noise = np.zeros(count, dtype=float)
    target = int(np.argmin(np.abs(geometry.zmid - MIN_BULK_TOP_Z_M)))
    noise[target] = 1.0
    noise_metrics = classify_frame(geometry, FrameFields(noise, velocity))
    assert not noise_metrics["active_raw"]


def _self_test_ascii_vtu_reader() -> None:
    x0, x1 = -D_M / 2.0, D_M / 2.0
    y0, y1 = 0.0, EXTRUSION_THICKNESS_M
    z0, z1 = RIM_Z_M, LAUNCH_BAND_TOP_M
    points = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    )
    point_text = " ".join(str(value) for point in points for value in point)
    raw = f"""<?xml version='1.0'?>
<VTKFile type='UnstructuredGrid'>
  <UnstructuredGrid><Piece NumberOfPoints='8' NumberOfCells='1'>
    <Points><DataArray Name='Points' NumberOfComponents='3' format='ascii'>{point_text}</DataArray></Points>
    <Cells>
      <DataArray Name='connectivity' format='ascii'>0 1 2 3 4 5 6 7</DataArray>
      <DataArray Name='offsets' format='ascii'>8</DataArray>
      <DataArray Name='types' format='ascii'>12</DataArray>
    </Cells>
    <CellData>
      <DataArray Name='alpha.water' format='ascii'>1</DataArray>
      <DataArray Name='U' NumberOfComponents='3' format='ascii'>0 0 1</DataArray>
    </CellData>
  </Piece></UnstructuredGrid>
</VTKFile>
"""
    with tempfile.TemporaryDirectory(prefix="eruption-gate-selftest-") as temp:
        path = Path(temp) / "internal.vtu"
        path.write_text(raw, encoding="utf-8")
        geometry, fields = read_geometry_and_fields(path)
    assert len(geometry.volumes) == 1
    assert math.isclose(float(fields.alpha[0]), 1.0)
    assert fields.velocity is not None and math.isclose(float(fields.velocity[0, 2]), 1.0)


def run_self_test() -> None:
    # Actual external-region vertical spacings: refined keeps D/16 vertically.
    _self_test_level(8, (1.30 - 1.02) / 44.0)
    _self_test_level(16, (1.30 - 1.02) / 176.0)
    _self_test_level(32, (1.30 - 1.02) / 176.0)
    _self_test_ascii_vtu_reader()

    rows = [
        {"sample_index": 0, "target_time_s": 0.0, "active_raw": False},
        {"sample_index": 1, "target_time_s": 0.1, "active_raw": True},
        {"sample_index": 2, "target_time_s": 0.2, "active_raw": True},
    ]
    apply_persistence(rows)
    assert not rows[0]["active_persistent"]
    assert rows[1]["active_persistent"] and rows[2]["active_persistent"]

    frames = [FrameRef(time, Path(f"frame-{index}")) for index, time in enumerate([3.0, 3.01, 3.1, 3.2])]
    selected, missing = select_common_time_frames(frames, 3.0)
    assert [round(target, 8) for _, target, _ in selected] == [3.0, 3.1, 3.2]
    assert not missing
    print("SELF_TEST_PASS: fixed physical gate, connectivity, and 0.10 s persistence")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", nargs="?", type=Path, help="OpenFOAM mesh-level case")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--vtk-dir",
        type=Path,
        help="existing foamToVTK ASCII output containing one *.vtm.series",
    )
    source.add_argument(
        "--export",
        action="store_true",
        help="run foamToVTK into a temporary directory, then delete the VTK export",
    )
    parser.add_argument(
        "--stage2-start",
        type=float,
        help="Stage-2 start time; default reads case/STAGE1_ACCEPTED_TIME",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: CASE/postProcessing/eruptionGate",
    )
    parser.add_argument(
        "--no-marker",
        action="store_true",
        help="write CSV/JSON but do not update the three case-root status markers",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic coarse/medium/refined tests; reads/writes no case",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if args.case_dir is None:
        raise SystemExit("case_dir is required unless --self-test is used")
    case_dir = args.case_dir.resolve()
    if not case_dir.is_dir():
        raise SystemExit(f"case directory does not exist: {case_dir}")
    stage2_start = _read_stage2_start(case_dir, args.stage2_start)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else case_dir / "postProcessing" / "eruptionGate"
    )

    if args.export:
        with tempfile.TemporaryDirectory(prefix="eruption-gate-vtk-") as temp:
            vtk_dir = Path(temp)
            export_ascii_vtk(case_dir, stage2_start, vtk_dir)
            summary = run_classifier(
                case_dir,
                vtk_dir,
                output_dir,
                stage2_start,
                write_marker=not args.no_marker,
            )
    else:
        vtk_dir = (
            args.vtk_dir.resolve()
            if args.vtk_dir is not None
            else case_dir / "VTK_ERUPTION_GATE"
        )
        if not vtk_dir.is_dir():
            raise SystemExit(
                f"VTK directory does not exist: {vtk_dir}; pass --vtk-dir or --export"
            )
        summary = run_classifier(
            case_dir,
            vtk_dir,
            output_dir,
            stage2_start,
            write_marker=not args.no_marker,
        )
    classification = summary["classification"]
    print(f"{classification['status']}: {classification['reason']}")
    print(f"outputs: {output_dir}")
    return 0 if classification["status"] == "PASS_ERUPTION" else 2


if __name__ == "__main__":
    sys.exit(main())
