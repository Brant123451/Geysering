#!/usr/bin/env python3
"""Definition-driven common observables for the Mahyawansi S1 comparison.

This module deliberately owns no OpenFOAM result marker.  It reads cell-centred
ASCII VTK evidence and probe files, records proxy/unavailable semantics, and
emits ordinary CSV/JSON artifacts below a caller-selected output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


class EvidenceError(RuntimeError):
    """Raised when an input cannot support the requested observable."""


@dataclass(frozen=True)
class FrameRef:
    time_s: float
    path: Path


@dataclass(frozen=True)
class Definitions:
    path: Path
    data: dict[str, Any]
    result_path: Path
    result: dict[str, Any]
    common_path: Path
    common: dict[str, Any]

    @property
    def dt(self) -> float:
        return float(self.common["comparison_time_grid_s"])

    @property
    def canonical(self) -> list[str]:
        return [str(item) for item in self.common["canonical_series"]]


@dataclass(frozen=True)
class Geometry:
    points: np.ndarray
    cells: np.ndarray
    xmin: np.ndarray
    xmax: np.ndarray
    ymin: np.ndarray
    ymax: np.ndarray
    zmin: np.ndarray
    zmax: np.ndarray
    xmid: np.ndarray
    zmid: np.ndarray
    volumes: np.ndarray
    adjacency: tuple[tuple[int, ...], ...]
    horizontal: np.ndarray
    supply: np.ndarray
    supply_top: np.ndarray
    riser: np.ndarray
    external: np.ndarray
    eruption_launch_band: np.ndarray
    riser_arrival_band: np.ndarray


@dataclass(frozen=True)
class FrameFields:
    alpha: np.ndarray
    velocity: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"missing YAML input: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"{path}: YAML root must be a mapping")
    return payload


def load_definitions(path: Path) -> Definitions:
    path = path.resolve()
    own = _load_yaml(path)
    authoritative = own.get("authoritative_inputs")
    if not isinstance(authoritative, dict):
        raise EvidenceError("definitions omit authoritative_inputs")
    result_path = (path.parent / str(authoritative["result_acceptance"])).resolve()
    common_path = (path.parent / str(authoritative["common_observables"])).resolve()
    result = _load_yaml(result_path)
    common = _load_yaml(common_path)

    pdf_path = (path.parent / str(authoritative["source_pdf"])).resolve()
    expected_pdf_hash = str(authoritative["source_pdf_sha256"]).upper()
    if not pdf_path.is_file() or sha256_file(pdf_path) != expected_pdf_hash:
        raise EvidenceError(
            "the directly reviewed Mahyawansi source PDF is missing or hash-mismatched"
        )
    if common.get("time_origin") != "stage_2_air_opening":
        raise EvidenceError("COMMON_OBSERVABLES time origin is not Stage-2 opening")
    if common.get("time_shift_allowed") is not False:
        raise EvidenceError("COMMON_OBSERVABLES must forbid time shifting")
    own_comparison = own.get("comparison", {})
    if own_comparison.get("time_shift_allowed") is not False:
        raise EvidenceError("observable definitions must forbid time shifting")
    common_dt = float(common["comparison_time_grid_s"])
    if not math.isclose(
        common_dt, float(own_comparison["common_grid_s"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise EvidenceError("comparison grid conflicts with COMMON_OBSERVABLES")
    if result.get("hard_physics_gate", {}).get("expected_eruption") is not True:
        raise EvidenceError("RESULT_ACCEPTANCE no longer requires eruption")
    event_gate = result["hard_physics_gate"]["fixed_physical_event_test"]
    result_dt = float(event_gate["comparison_sampling_interval_s"])
    if not math.isclose(common_dt, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError("the frozen common comparison grid must remain 0.10 s")
    if not math.isclose(result_dt, common_dt, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError("RESULT_ACCEPTANCE sampling interval conflicts with the common grid")
    if event_gate.get("connectivity") != "shared_cell_face":
        raise EvidenceError("the frozen eruption criterion must use shared-cell-face connectivity")
    if not math.isclose(float(event_gate["bulk_top_water_quantile"]), 0.99, abs_tol=1e-12):
        raise EvidenceError("the frozen eruption bulk-top statistic must remain q99")
    if not math.isclose(
        float(event_gate["minimum_continuous_duration_s"]), 0.10, abs_tol=1e-12
    ):
        raise EvidenceError("the frozen eruption persistence must remain 0.10 s")
    planned = float(result["hard_physics_gate"]["observation_window"]["planned_stage2_duration_s"])
    if not math.isclose(
        float(own_comparison["no_event_final_decision_time_s"]), planned, abs_tol=1e-12
    ) or not math.isclose(planned, 25.0, abs_tol=1e-12):
        raise EvidenceError("the complete no-event decision must remain at Stage-2 t=25 s")

    geom = own.get("geometry", {})
    evidence_status = geom.get("evidence_status", {})
    geometry_fields = [key for key in geom if key != "evidence_status"]
    if set(geometry_fields) != set(evidence_status):
        raise EvidenceError("every comparison geometry value must have evidence_status")
    result_rim = float(result["hard_physics_gate"]["riser_rim_z_m"])
    if not math.isclose(float(geom["riser_rim_z_m"]), result_rim, abs_tol=1e-12):
        raise EvidenceError("riser rim conflicts with RESULT_ACCEPTANCE")
    result_alpha = float(result["hard_physics_gate"]["water_phase_threshold"])
    if not math.isclose(
        float(own["field_extraction"]["water_wet_threshold"]),
        result_alpha,
        abs_tol=1e-12,
    ):
        raise EvidenceError("wet threshold conflicts with RESULT_ACCEPTANCE")
    d = float(geom["pipe_diameter_m"])
    launch_half_width = float(
        result["hard_physics_gate"]["fixed_physical_event_test"]["launch_band"]
        ["abs_x_max_m"]
    )
    if not math.isclose(launch_half_width, 0.5 * d, abs_tol=1e-12):
        raise EvidenceError("pipe diameter conflicts with frozen eruption launch band")

    required = {
        "time_s",
        "P1_gauge_Pa",
        "P2_gauge_Pa",
        "P3_gauge_Pa",
        "horizontal_gas_nose_x_m",
        "horizontal_gas_tail_x_m",
        "horizontal_gas_centroid_x_m",
        "horizontal_gas_volume_m3",
        "horizontal_slug_nose_x_m",
        "horizontal_slug_tail_x_m",
        "horizontal_slug_velocity_m_s",
        "gas_arrival_at_riser",
        "riser_connected_water_top_z_m",
        "riser_upward_liquid_flow_m3_s",
        "riser_downward_liquid_flow_m3_s",
        "mouth_liquid_outflow_m3_s",
        "internal_mouth_event_active",
    }
    missing = required - set(str(item) for item in common["canonical_series"])
    if missing:
        raise EvidenceError(f"COMMON_OBSERVABLES omits required fields: {sorted(missing)}")
    semantics = own["source_semantics"]["fig8_velocity"]
    if semantics.get("quantity") != (
        "water_velocity_magnitude_in_unmixed_middle_part_of_horizontal_slug"
    ):
        raise EvidenceError("Fig.8 velocity semantics changed or are ambiguous")
    forbidden = set(str(item) for item in semantics.get("forbidden_aliases", []))
    if "gas_nose_velocity" not in forbidden:
        raise EvidenceError("Fig.8 must explicitly forbid gas-nose aliasing")
    source_slug_target = result["published_and_figure_read_targets"]
    source_slug_target = source_slug_target["horizontal_slug_velocity_m_per_s"]["target"]
    if [float(item) for item in semantics["target_m_s"]] != [
        float(item) for item in source_slug_target
    ]:
        raise EvidenceError("Fig.8 target conflicts with RESULT_ACCEPTANCE")
    if (
        own["field_extraction"].get("published_slug_PIV_window_evidence_status")
        != "published_Figures_4_to_7_captions"
    ):
        raise EvidenceError("the published slug/PIV window lacks its source classification")
    published_mapping = own["source_semantics"]["probes"]["mapping_to_model_xz"]
    common_locations = common["fixed_probe_locations_xz_m"]
    common_aliases = {
        "P1": "P1",
        "P2": "P2",
        "P3": "P3",
        "P4": "H_upstream",
        "P5": "riser_left",
        "P6": "riser_right",
    }
    published_probe_coordinates = {
        "P1": (0.0, 0.0),
        "P2": (0.0, 0.30),
        "P3": (0.0, 0.45),
        "P4": (-0.80, 0.0),
        "P5": (-0.10, 0.0),
        "P6": (0.10, 0.0),
    }
    for paper_name, common_name in common_aliases.items():
        if not np.allclose(
            published_mapping[paper_name]["paper"],
            published_probe_coordinates[paper_name],
            atol=1e-12,
            rtol=0.0,
        ):
            raise EvidenceError(f"published {paper_name} paper coordinate changed")
        if not np.allclose(
            published_mapping[paper_name]["model"],
            published_probe_coordinates[paper_name],
            atol=1e-12,
            rtol=0.0,
        ):
            raise EvidenceError(f"published {paper_name} coordinate changed")
        if not np.allclose(
            published_mapping[paper_name]["model"],
            common_locations[common_name],
            atol=1e-12,
            rtol=0.0,
        ):
            raise EvidenceError(
                f"COMMON_OBSERVABLES {common_name} conflicts with published {paper_name}"
            )
    required_event_fields = {
        "target_time_s",
        "actual_time_s",
        "stage2_elapsed_s",
        "launch_component_count",
        "component_water_volume_m3",
        "volume_over_minimum",
        "bulk_top_q99_z_m",
        "bulk_height_above_rim_m",
        "active_raw",
        "active_persistent",
        "water_weighted_uz_m_per_s",
    }
    declared_event_fields = set(
        str(item)
        for item in result["required_acceptance_outputs"]["time_series_fields"]
    )
    if declared_event_fields != required_event_fields:
        raise EvidenceError("RESULT_ACCEPTANCE eruption time-series contract changed")
    return Definitions(path, own, result_path, result, common_path, common)


def _data_array(
    raw: str,
    name: str,
    *,
    scope: str | None = None,
    required: bool = True,
) -> np.ndarray | None:
    text = raw
    if scope is not None:
        block = re.search(rf"<{scope}\b[^>]*>(.*?)</{scope}>", text, re.S | re.I)
        if not block:
            if required:
                raise EvidenceError(f"missing <{scope}> block")
            return None
        text = block.group(1)
    match = re.search(
        rf"<DataArray\b(?P<a>[^>]*)\bName=['\"]{re.escape(name)}['\"]"
        rf"(?P<b>[^>]*)>(?P<body>.*?)</DataArray>",
        text,
        re.S | re.I,
    )
    if not match:
        if required:
            raise EvidenceError(f"missing DataArray {name!r}")
        return None
    attributes = match.group("a") + match.group("b")
    if not re.search(r"\bformat=['\"]ascii['\"]", attributes, re.I):
        raise EvidenceError(
            f"DataArray {name!r} is not ASCII; use foamToVTK -ascii -no-point-data"
        )
    values = np.fromstring(match.group("body"), sep=" ", dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise EvidenceError(f"DataArray {name!r} contains NaN/Inf")
    return values


def _hex_faces(vertices: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    if len(vertices) != 8:
        raise EvidenceError("only VTK_HEXAHEDRON cells are supported")
    v = tuple(int(item) for item in vertices)
    return (
        (v[0], v[1], v[2], v[3]),
        (v[4], v[5], v[6], v[7]),
        (v[0], v[1], v[5], v[4]),
        (v[1], v[2], v[6], v[5]),
        (v[2], v[3], v[7], v[6]),
        (v[3], v[0], v[4], v[7]),
    )


def build_adjacency(cells: np.ndarray) -> tuple[tuple[int, ...], ...]:
    neighbours: list[list[int]] = [[] for _ in range(len(cells))]
    owners: dict[tuple[int, ...], int] = {}
    for cell_index, vertices in enumerate(cells):
        for face in _hex_faces(vertices):
            key = tuple(sorted(face))
            other = owners.pop(key, None)
            if other is None:
                owners[key] = cell_index
            else:
                neighbours[cell_index].append(other)
                neighbours[other].append(cell_index)
    return tuple(tuple(sorted(items)) for items in neighbours)


def read_vtu(path: Path, defs: Definitions) -> tuple[Geometry, FrameFields]:
    raw = path.read_text(encoding="utf-8")
    points_raw = _data_array(raw, "Points")
    connectivity_raw = _data_array(raw, "connectivity")
    offsets_raw = _data_array(raw, "offsets")
    types_raw = _data_array(raw, "types")
    alpha_raw = _data_array(raw, "alpha.water", scope="CellData")
    velocity_raw = _data_array(raw, "U", scope="CellData")
    assert points_raw is not None
    assert connectivity_raw is not None
    assert offsets_raw is not None
    assert types_raw is not None
    assert alpha_raw is not None
    assert velocity_raw is not None
    points = points_raw.reshape(-1, 3)
    offsets = offsets_raw.astype(np.int64)
    counts = np.diff(np.r_[0, offsets])
    types = types_raw.astype(np.int64)
    if not np.all(counts == 8) or not np.all(types == 12):
        raise EvidenceError("only eight-node VTK_HEXAHEDRON cells are supported")
    cells = connectivity_raw.astype(np.int64).reshape(-1, 8)
    if len(alpha_raw) != len(cells) or len(velocity_raw) != 3 * len(cells):
        raise EvidenceError("cell-field length does not match VTK cell count")
    if np.any(alpha_raw < -1e-8) or np.any(alpha_raw > 1.0 + 1e-8):
        raise EvidenceError("alpha.water lies outside [0,1]")
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
    if np.any(volumes <= 0.0) or np.any(~np.isfinite(volumes)):
        raise EvidenceError("VTK contains non-positive/non-finite cell volumes")

    g = defs.data["geometry"]
    d = float(g["pipe_diameter_m"])
    half_d = 0.5 * d
    tol = 1e-9
    horizontal = (
        (xmid >= float(g["horizontal_x_min_m"]) - tol)
        & (xmid <= float(g["horizontal_x_max_m"]) + tol)
        & (zmin >= -half_d - tol)
        & (zmax <= half_d + tol)
    )
    supply = (
        (np.abs(xmid - float(g["air_inlet_x_m"])) <= half_d + tol)
        & (zmin >= half_d - tol)
        & (zmax <= float(g["air_stub_top_z_m"]) + tol)
    )
    supply_top = supply & (zmax >= float(g["air_stub_top_z_m"]) - tol)
    riser = (
        (np.abs(xmid - float(g["riser_x_m"])) <= half_d + tol)
        & (zmin >= half_d - tol)
        & (zmax <= float(g["riser_rim_z_m"]) + tol)
    )
    external = (
        (np.abs(xmid - float(g["riser_x_m"]))
         <= float(g["external_half_width_m"]) + tol)
        & (zmin >= float(g["riser_rim_z_m"]) - tol)
    )
    event = defs.result["hard_physics_gate"]["fixed_physical_event_test"]
    band = event["launch_band"]
    eruption_launch_band = (
        external
        & (np.abs(xmid) <= float(band["abs_x_max_m"]) + tol)
        & (zmid >= float(band["z_min_m"]) - tol)
        & (zmid <= float(band["z_max_m"]) + tol)
    )
    riser_arrival_band = horizontal & (np.abs(xmid) <= half_d + tol)
    for label, mask in {
        "horizontal main": horizontal,
        "supply branch": supply,
        "riser": riser,
        "external plume": external,
    }.items():
        if not np.any(mask):
            raise EvidenceError(f"VTK geometry contains no {label} cells")
    if not np.any(supply_top):
        raise EvidenceError("VTK geometry contains no cells at the air-inlet top")
    if not np.any(eruption_launch_band):
        raise EvidenceError("VTK geometry contains no frozen eruption launch band")
    geometry = Geometry(
        points=points,
        cells=cells,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        xmid=xmid,
        zmid=zmid,
        volumes=volumes,
        adjacency=build_adjacency(cells),
        horizontal=horizontal,
        supply=supply,
        supply_top=supply_top,
        riser=riser,
        external=external,
        eruption_launch_band=eruption_launch_band,
        riser_arrival_band=riser_arrival_band,
    )
    fields = FrameFields(alpha_raw, velocity_raw.reshape(-1, 3))
    return geometry, fields


def read_fields_only(path: Path, geometry: Geometry) -> FrameFields:
    raw = path.read_text(encoding="utf-8")
    alpha = _data_array(raw, "alpha.water", scope="CellData")
    velocity = _data_array(raw, "U", scope="CellData")
    assert alpha is not None and velocity is not None
    if len(alpha) != len(geometry.cells) or len(velocity) != 3 * len(geometry.cells):
        raise EvidenceError(f"{path}: field length differs from first VTK frame")
    if np.any(alpha < -1e-8) or np.any(alpha > 1.0 + 1e-8):
        raise EvidenceError(f"{path}: alpha.water lies outside [0,1]")
    return FrameFields(alpha, velocity.reshape(-1, 3))


def _internal_vtu(vtm: Path) -> Path:
    raw = vtm.read_text(encoding="utf-8")
    match = re.search(
        r"<DataSet\b[^>]*\bname=['\"]internal['\"][^>]*\bfile=['\"]([^'\"]+)['\"]",
        raw,
        re.I,
    )
    if not match:
        match = re.search(
            r"<DataSet\b[^>]*\bfile=['\"]([^'\"]*internal\.vtu)['\"]", raw, re.I
        )
    if not match:
        raise EvidenceError(f"{vtm}: no internal.vtu reference")
    result = (vtm.parent / match.group(1)).resolve()
    if not result.is_file():
        raise EvidenceError(f"{vtm}: referenced internal VTU is missing: {result}")
    return result


def find_vtk_frames(vtk_dir: Path) -> tuple[Path, list[FrameRef]]:
    series = sorted(vtk_dir.resolve().rglob("*.vtm.series"))
    if len(series) != 1:
        raise EvidenceError(
            f"expected one *.vtm.series below {vtk_dir}, found {len(series)}"
        )
    payload = json.loads(series[0].read_text(encoding="utf-8"))
    frames: list[FrameRef] = []
    for entry in payload.get("files", []):
        vtm = (series[0].parent / str(entry["name"])).resolve()
        if not vtm.is_file():
            raise EvidenceError(f"missing VTM series entry: {vtm}")
        frames.append(FrameRef(float(entry["time"]), _internal_vtu(vtm)))
    frames.sort(key=lambda item: item.time_s)
    if not frames:
        raise EvidenceError(f"{series[0]} contains no frames")
    if any(b.time_s <= a.time_s for a, b in zip(frames, frames[1:])):
        raise EvidenceError("VTK frame times must be strictly increasing")
    return series[0], frames


def select_common_frames(
    frames: Sequence[FrameRef], stage2_start_s: float, defs: Definitions
) -> tuple[list[tuple[int, float, FrameRef]], list[float]]:
    dt = defs.dt
    available = [item for item in frames if item.time_s >= stage2_start_s - 1e-7]
    if not available:
        raise EvidenceError("no VTK frame at or after Stage-2 opening")
    last_index = int(math.floor((available[-1].time_s - stage2_start_s) / dt + 1e-7))
    tolerance = (
        float(defs.data["field_extraction"]["frame_to_common_time_tolerance_fraction"])
        * dt
    )
    selected: list[tuple[int, float, FrameRef]] = []
    missing: list[float] = []
    used: set[Path] = set()
    for index in range(last_index + 1):
        target_abs = stage2_start_s + index * dt
        nearest = min(available, key=lambda item: abs(item.time_s - target_abs))
        if abs(nearest.time_s - target_abs) > tolerance or nearest.path in used:
            missing.append(index * dt)
            continue
        selected.append((index, index * dt, nearest))
        used.add(nearest.path)
    if not selected or not math.isclose(selected[0][1], 0.0, abs_tol=1e-12):
        raise EvidenceError("the common series must include unshifted Stage-2 t=0")
    return selected, missing


def _components(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> list[np.ndarray]:
    visited = np.zeros(len(mask), dtype=bool)
    result: list[np.ndarray] = []
    for seed in np.flatnonzero(mask):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [int(seed)]
        members: list[int] = []
        while stack:
            cell = stack.pop()
            members.append(cell)
            for neighbour in adjacency[cell]:
                if mask[neighbour] and not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(int(neighbour))
        result.append(np.asarray(members, dtype=np.int64))
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or len(values) != len(weights):
        raise EvidenceError("weighted quantile requires equal non-empty arrays")
    total = float(weights.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise EvidenceError("weighted quantile weights are not positive and finite")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * total, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def _supply_connected_gas(
    geometry: Geometry, fields: FrameFields, defs: Definitions
) -> np.ndarray:
    alpha_max = float(defs.data["field_extraction"]["gas_cell_max_water_fraction"])
    domain = geometry.horizontal | geometry.supply
    gas = domain & (fields.alpha <= alpha_max)
    candidates = [
        component
        for component in _components(gas, geometry.adjacency)
        if np.any(geometry.supply_top[component])
    ]
    if not candidates:
        return np.asarray([], dtype=np.int64)
    return max(
        candidates,
        key=lambda cells: float(
            np.dot(1.0 - fields.alpha[cells], geometry.volumes[cells])
        ),
    )


def _horizontal_columns(
    geometry: Geometry, fields: FrameFields
) -> list[dict[str, float | np.ndarray]]:
    groups: dict[tuple[float, float], list[int]] = {}
    for index in np.flatnonzero(geometry.horizontal):
        key = (round(float(geometry.xmin[index]), 12), round(float(geometry.xmax[index]), 12))
        groups.setdefault(key, []).append(int(index))
    columns: list[dict[str, float | np.ndarray]] = []
    for (xmin, xmax), members in sorted(groups.items()):
        cells = np.asarray(members, dtype=np.int64)
        volume = float(geometry.volumes[cells].sum())
        water_volume = float(np.dot(fields.alpha[cells], geometry.volumes[cells]))
        columns.append(
            {
                "xmin": xmin,
                "xmax": xmax,
                "xmid": 0.5 * (xmin + xmax),
                "cells": cells,
                "water_holdup": water_volume / volume,
            }
        )
    return columns


def _detect_slug(
    geometry: Geometry, fields: FrameFields, defs: Definitions
) -> tuple[float, float, np.ndarray]:
    cfg = defs.data["field_extraction"]
    full_min = float(cfg["full_bore_column_min_water_holdup"])
    gas_min = float(cfg["surrounding_gas_min_column_fraction"])
    min_length = float(cfg["slug_min_axial_length_m"])
    search_min, search_max = (float(v) for v in cfg["slug_search_x_m"])
    piv_min, piv_max = (float(v) for v in cfg["published_slug_PIV_window_x_m"])
    anchor = 0.5 * (piv_min + piv_max)
    columns = [
        item
        for item in _horizontal_columns(geometry, fields)
        if search_min <= float(item["xmid"]) <= search_max
    ]
    if len(columns) < 3:
        return math.nan, math.nan, np.asarray([], dtype=np.int64)
    full = np.asarray([float(item["water_holdup"]) >= full_min for item in columns])
    gas_bearing = np.asarray(
        [1.0 - float(item["water_holdup"]) >= gas_min for item in columns]
    )
    candidates: list[tuple[float, float, np.ndarray]] = []
    start = 0
    while start < len(columns):
        if not full[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(columns) and full[end + 1]:
            end += 1
        if start > 0 and end + 1 < len(columns) and gas_bearing[start - 1] and gas_bearing[end + 1]:
            tail = float(columns[start]["xmin"])
            nose = float(columns[end]["xmax"])
            if nose - tail >= min_length - 1e-12:
                cells = np.concatenate(
                    [np.asarray(columns[index]["cells"], dtype=np.int64) for index in range(start, end + 1)]
                )
                candidates.append((tail, nose, cells))
        start = end + 1
    if not candidates:
        return math.nan, math.nan, np.asarray([], dtype=np.int64)
    return min(candidates, key=lambda item: abs(0.5 * (item[0] + item[1]) - anchor))


def _section_flux_proxy(
    geometry: Geometry,
    fields: FrameFields,
    *,
    centre_x: float,
    half_width: float,
    z_section: float,
    direction: str = "vertical",
) -> tuple[float, float, float, float]:
    """Return liquid up/down and gas up/down volume-flux proxies.

    One cell immediately below the requested section is selected in each
    transverse x-column.  Cell-centred velocity is multiplied by projected
    face area; this is not an OpenFOAM face flux and remains explicitly a
    proxy.
    """

    candidate = np.flatnonzero(
        (np.abs(geometry.xmid - centre_x) <= half_width + 1e-9)
        & (geometry.zmid < z_section + 1e-10)
        & (geometry.zmax <= z_section + 1e-8)
    )
    if len(candidate) == 0:
        return (math.nan,) * 4
    groups: dict[tuple[float, float], list[int]] = {}
    for cell in candidate:
        key = (round(float(geometry.xmin[cell]), 12), round(float(geometry.xmax[cell]), 12))
        groups.setdefault(key, []).append(int(cell))
    selected: list[int] = []
    for cells in groups.values():
        selected.append(max(cells, key=lambda index: float(geometry.zmax[index])))
    cells = np.asarray(selected, dtype=np.int64)
    distance = z_section - geometry.zmax[cells]
    local_dz = geometry.zmax[cells] - geometry.zmin[cells]
    keep = distance <= 1.01 * local_dz + 1e-9
    cells = cells[keep]
    if len(cells) == 0:
        return (math.nan,) * 4
    if direction != "vertical":
        raise EvidenceError(f"unsupported section direction: {direction}")
    area = (geometry.xmax[cells] - geometry.xmin[cells]) * (
        geometry.ymax[cells] - geometry.ymin[cells]
    )
    uz = fields.velocity[cells, 2]
    alpha = fields.alpha[cells]
    liquid_up = float(np.dot(alpha * np.maximum(uz, 0.0), area))
    liquid_down = float(np.dot(alpha * np.maximum(-uz, 0.0), area))
    gas = 1.0 - alpha
    gas_up = float(np.dot(gas * np.maximum(uz, 0.0), area))
    gas_down = float(np.dot(gas * np.maximum(-uz, 0.0), area))
    return liquid_up, liquid_down, gas_up, gas_down


def _largest_riser_wet_component(
    geometry: Geometry, fields: FrameFields, defs: Definitions
) -> np.ndarray:
    wet_threshold = float(defs.data["field_extraction"]["water_wet_threshold"])
    wet = (geometry.riser | geometry.external) & (fields.alpha >= wet_threshold)
    candidates = [
        component
        for component in _components(wet, geometry.adjacency)
        if np.any(geometry.riser[component])
    ]
    if not candidates:
        return np.asarray([], dtype=np.int64)
    return max(
        candidates,
        key=lambda cells: float(np.dot(fields.alpha[cells], geometry.volumes[cells])),
    )


def _eruption_raw(
    geometry: Geometry, fields: FrameFields, defs: Definitions
) -> dict[str, float | int | bool]:
    threshold = float(defs.data["field_extraction"]["water_wet_threshold"])
    wet = geometry.external & (fields.alpha >= threshold)
    candidates = [
        component
        for component in _components(wet, geometry.adjacency)
        if np.any(geometry.eruption_launch_band[component])
    ]
    if not candidates:
        return {
            "launch_component_count": 0,
            "component_water_volume_m3": 0.0,
            "bulk_top_q99_z_m": math.nan,
            "active_raw": False,
            "water_weighted_uz_m_per_s": math.nan,
        }
    gate = defs.result["hard_physics_gate"]["fixed_physical_event_test"]
    min_volume = float(gate["minimum_connected_water_volume_m3"])
    min_top = float(gate["minimum_connected_bulk_top_z_m"])
    quantile = float(gate["bulk_top_water_quantile"])
    evaluated: list[dict[str, float | bool]] = []
    for cells in candidates:
        weights = fields.alpha[cells] * geometry.volumes[cells]
        volume = float(weights.sum())
        top = _weighted_quantile(geometry.zmax[cells], weights, quantile)
        mean_uz = float(np.dot(weights, fields.velocity[cells, 2]) / volume)
        evaluated.append(
            {
                "active_raw": volume >= min_volume - 1e-12 and top >= min_top - 1e-12,
                "component_water_volume_m3": volume,
                "bulk_top_q99_z_m": top,
                "water_weighted_uz_m_per_s": mean_uz,
            }
        )
    chosen = max(
        evaluated,
        key=lambda item: (
            bool(item["active_raw"]),
            float(item["component_water_volume_m3"]),
        ),
    )
    return {"launch_component_count": len(candidates), **chosen}


def extract_frame(
    geometry: Geometry, fields: FrameFields, defs: Definitions
) -> dict[str, float | bool]:
    gcfg = defs.data["geometry"]
    fcfg = defs.data["field_extraction"]
    d = float(gcfg["pipe_diameter_m"])
    gas_cells = _supply_connected_gas(geometry, fields, defs)
    horizontal_gas = gas_cells[geometry.horizontal[gas_cells]] if len(gas_cells) else gas_cells
    if len(horizontal_gas):
        weights = (1.0 - fields.alpha[horizontal_gas]) * geometry.volumes[horizontal_gas]
        gas_volume = float(weights.sum())
        gas_nose = float(geometry.xmax[horizontal_gas].max())
        gas_tail = float(geometry.xmin[horizontal_gas].min())
        gas_centroid = float(np.dot(weights, geometry.xmid[horizontal_gas]) / gas_volume)
        gas_arrival = bool(np.any(geometry.riser_arrival_band[horizontal_gas]))
    else:
        gas_volume = 0.0
        gas_nose = gas_tail = gas_centroid = math.nan
        gas_arrival = False
    supply_members = gas_cells[geometry.supply[gas_cells]] if len(gas_cells) else gas_cells
    supply_front = (
        float(geometry.zmin[supply_members].min()) if len(supply_members) else math.nan
    )

    slug_tail, slug_nose, slug_cells = _detect_slug(geometry, fields, defs)
    piv_min, piv_max = (float(v) for v in fcfg["published_slug_PIV_window_x_m"])
    unmixed = float(fcfg["unmixed_water_min_fraction"])
    piv_cells = np.flatnonzero(
        geometry.horizontal
        & (geometry.xmid >= piv_min - 1e-12)
        & (geometry.xmid <= piv_max + 1e-12)
        & (fields.alpha >= unmixed)
    )
    slug_intersects_piv = (
        math.isfinite(slug_tail)
        and math.isfinite(slug_nose)
        and slug_nose >= piv_min
        and slug_tail <= piv_max
    )
    if slug_intersects_piv and len(piv_cells):
        weights = fields.alpha[piv_cells] * geometry.volumes[piv_cells]
        speed = np.linalg.norm(fields.velocity[piv_cells], axis=1)
        slug_middle_water_speed = float(np.dot(weights, speed) / weights.sum())
    else:
        slug_middle_water_speed = math.nan

    riser_component = _largest_riser_wet_component(geometry, fields, defs)
    if len(riser_component):
        weights = fields.alpha[riser_component] * geometry.volumes[riser_component]
        riser_top = _weighted_quantile(
            geometry.zmax[riser_component],
            weights,
            float(fcfg["weighted_top_quantile"]),
        )
    else:
        riser_top = math.nan

    riser_up, riser_down, _, _ = _section_flux_proxy(
        geometry,
        fields,
        centre_x=float(gcfg["riser_x_m"]),
        half_width=0.5 * d,
        z_section=float(fcfg["riser_flux_section_z_m"]),
    )
    mouth_liquid_up, _, mouth_gas_up, _ = _section_flux_proxy(
        geometry,
        fields,
        centre_x=float(gcfg["riser_x_m"]),
        half_width=0.5 * d,
        z_section=float(gcfg["riser_rim_z_m"]),
    )
    supply_liquid_down = _section_flux_proxy(
        geometry,
        fields,
        centre_x=float(gcfg["air_inlet_x_m"]),
        half_width=0.5 * d,
        z_section=0.5 * d + min(
            geometry.zmax[geometry.supply] - geometry.zmin[geometry.supply]
        ),
    )[1]
    eruption = _eruption_raw(geometry, fields, defs)
    event_volume = float(eruption["component_water_volume_m3"])
    event_top = float(eruption["bulk_top_q99_z_m"])
    min_volume = float(
        defs.result["hard_physics_gate"]["fixed_physical_event_test"]
        ["minimum_connected_water_volume_m3"]
    )
    rim = float(gcfg["riser_rim_z_m"])
    return {
        "horizontal_gas_nose_x_m": gas_nose,
        "horizontal_gas_tail_x_m": gas_tail,
        "horizontal_gas_centroid_x_m": gas_centroid,
        "horizontal_gas_volume_m3": gas_volume,
        "gas_arrival_at_riser": gas_arrival,
        "supply_branch_gas_front_z_m": supply_front,
        "horizontal_slug_nose_x_m": slug_nose,
        "horizontal_slug_tail_x_m": slug_tail,
        # This canonical name is source-semantics constrained: water |U|, not edge speed.
        "horizontal_slug_velocity_m_s": slug_middle_water_speed,
        "riser_connected_water_top_z_m": riser_top,
        "riser_upward_liquid_flow_m3_s": riser_up,
        "riser_downward_liquid_flow_m3_s": riser_down,
        "mouth_liquid_outflow_m3_s": mouth_liquid_up,
        "supply_branch_liquid_outflow_m3_s": supply_liquid_down,
        "mouth_gas_volume_outflow_m3_s_proxy": mouth_gas_up,
        # Exact RESULT_ACCEPTANCE names. These are the shared-face-connected,
        # launch-attached external component diagnostics, not a rendered-image proxy.
        "launch_component_count": int(eruption["launch_component_count"]),
        "component_water_volume_m3": event_volume,
        "volume_over_minimum": event_volume / min_volume,
        "bulk_top_q99_z_m": event_top,
        "bulk_height_above_rim_m": event_top - rim if math.isfinite(event_top) else math.nan,
        "active_raw": bool(eruption["active_raw"]),
        "water_weighted_uz_m_per_s": float(eruption["water_weighted_uz_m_per_s"]),
    }


def apply_persistence(rows: list[dict[str, Any]], defs: Definitions) -> None:
    persistence = float(
        defs.result["hard_physics_gate"]["fixed_physical_event_test"]
        ["minimum_continuous_duration_s"]
    )
    for row in rows:
        row["internal_mouth_event_active"] = False
        row["active_persistent"] = False
        row["eruption_event_id"] = math.nan
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
            and int(rows[end + 1]["sample_index"]) == int(rows[end]["sample_index"]) + 1
        ):
            end += 1
        duration = float(rows[end]["time_s"]) - float(rows[start]["time_s"])
        if duration >= persistence - 1e-9:
            event_id += 1
            for index in range(start, end + 1):
                rows[index]["internal_mouth_event_active"] = True
                rows[index]["active_persistent"] = True
                rows[index]["eruption_event_id"] = event_id
        start = end + 1


def _finite_difference(rows: list[dict[str, Any]], source: str, target: str) -> None:
    for row in rows:
        row[target] = math.nan
    for index, row in enumerate(rows):
        candidates: list[int]
        if 0 < index < len(rows) - 1:
            candidates = [index - 1, index + 1]
        elif index + 1 < len(rows):
            candidates = [index, index + 1]
        elif index > 0:
            candidates = [index - 1, index]
        else:
            continue
        left, right = (rows[item] for item in candidates)
        a, b = float(left[source]), float(right[source])
        dt = float(right["time_s"]) - float(left["time_s"])
        if math.isfinite(a) and math.isfinite(b) and dt > 0.0:
            row[target] = (b - a) / dt


def _parse_probe_file(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, list[float]]]]:
    locations: dict[int, tuple[float, float]] = {}
    samples: list[tuple[float, list[float]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        header = re.match(
            r"^#\s*Probe\s+(\d+)\s*\(\s*([^\s]+)\s+([^\s]+)\s+([^\s]+)\s*\)",
            stripped,
        )
        if header:
            index = int(header.group(1))
            # OpenFOAM coordinates are (x, thin-y, z); compare x/z only.
            locations[index] = (float(header.group(2)), float(header.group(4)))
            continue
        if stripped.startswith("#"):
            continue
        values = [float(item) for item in FLOAT_RE.findall(stripped)]
        if len(values) < 2:
            raise EvidenceError(f"{path}:{line_number}: malformed scalar probe row")
        samples.append((values[0], values[1:]))
    if not locations:
        raise EvidenceError(f"{path}: no '# Probe N (...)' location headers")
    count = max(locations) + 1
    if sorted(locations) != list(range(count)):
        raise EvidenceError(f"{path}: non-contiguous probe indices")
    if any(len(values) != count for _, values in samples):
        raise EvidenceError(f"{path}: probe sample width does not match headers")
    return [locations[index] for index in range(count)], samples


def read_pressure_probes(
    case_dir: Path, defs: Definitions
) -> tuple[list[tuple[float, list[float]]], dict[str, Any]]:
    root = case_dir / "postProcessing" / "probesJHR"
    if not root.is_dir():
        raise EvidenceError(f"missing probesJHR output: {root}")
    segments: list[tuple[float, Path]] = []
    for child in root.iterdir():
        try:
            start = float(child.name)
        except ValueError:
            continue
        if (child / "p").is_file():
            segments.append((start, child / "p"))
    segments.sort(key=lambda item: (item[0], item[1].parent.name))
    if not segments:
        raise EvidenceError(f"no numeric probesJHR segments contain p below {root}")
    expected_locations: list[tuple[float, float]] | None = None
    merged: dict[float, tuple[float, list[float]]] = {}
    paths: list[str] = []
    for _, path in segments:
        locations, samples = _parse_probe_file(path)
        if expected_locations is None:
            expected_locations = locations
        elif not np.allclose(locations, expected_locations, atol=1e-9, rtol=0.0):
            raise EvidenceError("probe locations change across restart segments")
        for time_s, values in samples:
            merged[round(time_s, 12)] = (time_s, values)
        paths.append(str(path.resolve()))
    assert expected_locations is not None
    mappings = defs.data["source_semantics"]["probes"]["mapping_to_model_xz"]
    paper_names = ["P1", "P2", "P3", "P4", "P5", "P6"]
    if len(expected_locations) != len(paper_names):
        raise EvidenceError(
            f"expected six published probes P1-P6, found {len(expected_locations)}"
        )
    for index, name in enumerate(paper_names):
        expected = tuple(float(v) for v in mappings[name]["model"])
        if not np.allclose(expected_locations[index], expected, atol=1e-9, rtol=0.0):
            raise EvidenceError(
                f"probe {index} does not match published {name}: "
                f"{expected_locations[index]} != {expected}"
            )
    rows = sorted(merged.values(), key=lambda item: item[0])
    if len(rows) < 2:
        raise EvidenceError("pressure probe evidence has fewer than two samples")
    return rows, {
        "segments": paths,
        "locations_xz_m": [list(item) for item in expected_locations],
        "paper_probe_names": paper_names,
        "evidence_status": "published_Figure_2_caption",
        "project_output_aliases": [str(mappings[name]["output"]) for name in paper_names],
    }


def _interpolate_probe(
    samples: Sequence[tuple[float, list[float]]],
    target_abs_s: float,
    probe_index: int,
    max_gap_s: float,
) -> float:
    times = np.asarray([item[0] for item in samples], dtype=float)
    position = int(np.searchsorted(times, target_abs_s, side="left"))
    if position < len(times) and math.isclose(times[position], target_abs_s, abs_tol=1e-10):
        return float(samples[position][1][probe_index])
    if position == 0 or position == len(times):
        return math.nan
    left_t, left_values = samples[position - 1]
    right_t, right_values = samples[position]
    if right_t - left_t > max_gap_s + 1e-12:
        return math.nan
    fraction = (target_abs_s - left_t) / (right_t - left_t)
    return float(left_values[probe_index] + fraction * (right_values[probe_index] - left_values[probe_index]))


def read_stage2_start(case_dir: Path, explicit: float | None) -> float:
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value):
            raise EvidenceError("explicit Stage-2 opening time is not finite")
        return value
    marker = case_dir / "STAGE1_ACCEPTED_TIME"
    if not marker.is_file():
        raise EvidenceError(
            "Stage-2 opening is unknown; pass --stage2-start or provide STAGE1_ACCEPTED_TIME"
        )
    value = float(marker.read_text(encoding="utf-8").strip())
    if not math.isfinite(value):
        raise EvidenceError("STAGE1_ACCEPTED_TIME is not finite")
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return ""
    return value


def ensure_diagnostic_output_path(path: Path, comparison_root: Path) -> Path:
    resolved = path.resolve()
    root = comparison_root.resolve()
    if not resolved.is_relative_to(root):
        raise EvidenceError(
            f"output must stay below the independent comparison directory: {root}"
        )
    forbidden = {
        "RESULT_ACCEPTED",
        "ERUPTION_ACCEPTED",
        "STAGE2_COMPLETE_UNVALIDATED",
        "RUN_COMPLETE",
    }
    if any(part.upper() in forbidden for part in resolved.relative_to(root).parts):
        raise EvidenceError("diagnostic comparison tools cannot write solver acceptance markers")
    return resolved


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, math.nan)) for key in columns})


def extract_2d_series(
    *,
    case_dir: Path,
    vtk_dir: Path,
    stage2_start_s: float,
    mesh_level: str,
    defs: Definitions,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required_levels = [str(item) for item in defs.data["comparison"]["required_mesh_levels"]]
    if mesh_level not in required_levels:
        raise EvidenceError(f"unknown 2-D mesh level {mesh_level!r}; expected {required_levels}")
    series_path, frames = find_vtk_frames(vtk_dir)
    selected, missing_times = select_common_frames(frames, stage2_start_s, defs)
    geometry, first_fields = read_vtu(selected[0][2].path, defs)
    pressure_samples, pressure_meta = read_pressure_probes(case_dir, defs)
    pressure_reference = float(
        defs.data["field_extraction"]["pressure_absolute_reference_Pa"]
    )
    max_probe_gap = float(
        defs.data["field_extraction"]["probe_interpolation_max_gap_s"]
    )
    aliases = [
        defs.data["source_semantics"]["probes"]["mapping_to_model_xz"][name]["output"]
        for name in ("P1", "P2", "P3", "P4", "P5", "P6")
    ]
    rows: list[dict[str, Any]] = []
    vtk_hashes: dict[str, str] = {}
    for ordinal, (sample_index, elapsed_s, frame) in enumerate(selected):
        fields = first_fields if ordinal == 0 else read_fields_only(frame.path, geometry)
        frame_values = extract_frame(geometry, fields, defs)
        row: dict[str, Any] = {name: math.nan for name in defs.canonical}
        row.update(frame_values)
        row.update(
            {
                "sample_index": sample_index,
                "time_s": elapsed_s,
                "target_time_s": stage2_start_s + elapsed_s,
                "actual_time_s": frame.time_s,
                "stage2_elapsed_s": frame.time_s - stage2_start_s,
                "target_absolute_time_s": stage2_start_s + elapsed_s,
                "actual_absolute_time_s": frame.time_s,
                "vtk_time_error_s": frame.time_s - (stage2_start_s + elapsed_s),
            }
        )
        for probe_index, alias in enumerate(aliases):
            absolute_p = _interpolate_probe(
                pressure_samples,
                stage2_start_s + elapsed_s,
                probe_index,
                max_probe_gap,
            )
            row[str(alias)] = absolute_p - pressure_reference if math.isfinite(absolute_p) else math.nan
        rows.append(row)
        vtk_hashes[str(frame.path.resolve())] = sha256_file(frame.path)
    apply_persistence(rows, defs)
    _finite_difference(
        rows, "horizontal_gas_nose_x_m", "horizontal_gas_nose_speed_m_s_proxy"
    )
    _finite_difference(
        rows, "horizontal_slug_nose_x_m", "horizontal_slug_nose_speed_m_s_proxy"
    )
    cumulative = 0.0
    previous_time: float | None = None
    previous_flux: float | None = None
    for row in rows:
        flux = float(row["mouth_liquid_outflow_m3_s"])
        time_s = float(row["time_s"])
        if previous_time is not None and previous_flux is not None and math.isfinite(flux) and math.isfinite(previous_flux):
            cumulative += 0.5 * (flux + previous_flux) * (time_s - previous_time)
        row["cumulative_mouth_liquid_outflow_m3"] = cumulative
        previous_time, previous_flux = time_s, flux

    meta = {
        "schema_version": 1,
        "case_id": defs.data["case_id"],
        "physical_condition_count": 1,
        "mesh_level": mesh_level,
        "case_dir": str(case_dir.resolve()),
        "vtk_series": str(series_path.resolve()),
        "stage2_start_absolute_s": stage2_start_s,
        "time_origin": "stage_2_air_opening",
        "time_shift_applied_s": 0.0,
        "common_grid_s": defs.dt,
        "missing_common_times_s": missing_times,
        "definition_file": str(defs.path),
        "definition_sha256": sha256_file(defs.path),
        "result_acceptance_file": str(defs.result_path),
        "result_acceptance_sha256": sha256_file(defs.result_path),
        "common_observables_file": str(defs.common_path),
        "common_observables_sha256": sha256_file(defs.common_path),
        "source_pdf_semantics": defs.data["source_semantics"],
        "observable_status": defs.data["observable_status"],
        "categorical_mapping": defs.data["observable_status"]["categorical_mapping"],
        "pressure_probe_evidence": pressure_meta,
        "vtk_sha256": vtk_hashes,
        "result_marker_written": False,
    }
    return rows, meta


def two_d_columns(defs: Definitions) -> list[str]:
    extras = [
        "sample_index",
        "target_time_s",
        "actual_time_s",
        "stage2_elapsed_s",
        "target_absolute_time_s",
        "actual_absolute_time_s",
        "vtk_time_error_s",
        "horizontal_gas_nose_speed_m_s_proxy",
        "horizontal_slug_nose_speed_m_s_proxy",
        "mouth_gas_volume_outflow_m3_s_proxy",
        "launch_component_count",
        "component_water_volume_m3",
        "volume_over_minimum",
        "bulk_top_q99_z_m",
        "bulk_height_above_rim_m",
        "active_raw",
        "active_persistent",
        "water_weighted_uz_m_per_s",
        "eruption_event_id",
    ]
    canonical = list(defs.canonical)
    return canonical + [item for item in extras if item not in canonical]


def horizontal_columns() -> list[str]:
    return [
        "time_s",
        "horizontal_gas_nose_x_m",
        "horizontal_gas_tail_x_m",
        "horizontal_gas_centroid_x_m",
        "horizontal_gas_volume_m3",
        "horizontal_gas_nose_speed_m_s_proxy",
        "gas_arrival_at_riser",
        "supply_branch_gas_front_z_m",
        "horizontal_slug_nose_x_m",
        "horizontal_slug_tail_x_m",
        "horizontal_slug_velocity_m_s",
        "horizontal_slug_nose_speed_m_s_proxy",
    ]


def read_numeric_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    if not path.is_file():
        raise EvidenceError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise EvidenceError(f"{path}: CSV has no header")
        rows: list[dict[str, float]] = []
        for line_number, raw in enumerate(reader, 2):
            row: dict[str, float] = {}
            for name in reader.fieldnames:
                token = (raw.get(name) or "").strip()
                if token == "":
                    row[name] = math.nan
                    continue
                try:
                    value = float(token)
                except ValueError as exc:
                    raise EvidenceError(
                        f"{path}:{line_number}: {name} is not numeric: {token!r}"
                    ) from exc
                row[name] = value
            rows.append(row)
    if not rows:
        raise EvidenceError(f"{path}: CSV contains no rows")
    return list(reader.fieldnames), rows


def _metadata_path_for_2d_csv(path: Path) -> Path:
    if path.name == "2d_common_timeseries.csv":
        return path.with_name("2d_common_timeseries.metadata.json")
    return path.with_name(path.name + ".metadata.json")


def validate_2d_metadata(path: Path, level: str, defs: Definitions) -> dict[str, Any]:
    metadata_path = _metadata_path_for_2d_csv(path)
    if not metadata_path.is_file():
        raise EvidenceError(f"2-D {level}: missing extractor metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"2-D {level}: malformed extractor metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise EvidenceError(f"2-D {level}: extractor metadata root is not an object")
    exact = {
        "case_id": defs.data["case_id"],
        "mesh_level": level,
        "time_origin": "stage_2_air_opening",
        "time_shift_applied_s": 0.0,
        "common_grid_s": defs.dt,
        "definition_sha256": sha256_file(defs.path),
        "result_acceptance_sha256": sha256_file(defs.result_path),
        "common_observables_sha256": sha256_file(defs.common_path),
        "csv_sha256": sha256_file(path),
        "result_marker_written": False,
    }
    for name, expected in exact.items():
        if metadata.get(name) != expected:
            raise EvidenceError(
                f"2-D {level}: metadata {name}={metadata.get(name)!r} != {expected!r}"
            )
    if metadata.get("physical_condition_count") != 1:
        raise EvidenceError(f"2-D {level}: metadata must identify one physical condition")
    vtk_hashes = metadata.get("vtk_sha256")
    if not isinstance(vtk_hashes, dict) or not vtk_hashes:
        raise EvidenceError(f"2-D {level}: metadata has no hashed VTK frame evidence")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9A-F]{64}", value) is None
        for value in vtk_hashes.values()
    ):
        raise EvidenceError(f"2-D {level}: metadata contains malformed VTK hashes")
    return {
        "path": str(metadata_path.resolve()),
        "sha256": sha256_file(metadata_path),
        "mesh_level": level,
        "case_id": metadata["case_id"],
        "vtk_frame_count": len(vtk_hashes),
        "status": "extractor_provenance_and_mesh_identity_verified",
    }


def _validate_time_series(rows: Sequence[Mapping[str, float]], label: str) -> np.ndarray:
    if "time_s" not in rows[0]:
        raise EvidenceError(f"{label}: missing time_s")
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise EvidenceError(f"{label}: time_s must be finite and strictly increasing")
    if not math.isclose(float(times[0]), 0.0, rel_tol=0.0, abs_tol=1e-10):
        raise EvidenceError(f"{label}: Stage-2 series must begin at unshifted t=0")
    return times


def validate_2d_event_evidence(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, float]],
    defs: Definitions,
    label: str,
) -> dict[str, Any]:
    """Fail closed unless the 2-D branch flag is backed by the frozen spatial test.

    Connectivity itself is established upstream from the ASCII VTK shared-face graph.
    This gate verifies that the exported chosen launch-attached component satisfies the
    frozen volume/q99 predicate and that the persistent flag was derived from the raw
    predicate on consecutive 0.10-s samples.  A caller-supplied Boolean column alone is
    never accepted as 2-D eruption evidence.
    """

    required = {
        "sample_index",
        "time_s",
        *(
            str(item)
            for item in defs.result["required_acceptance_outputs"]["time_series_fields"]
        ),
        "internal_mouth_event_active",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise EvidenceError(f"{label}: missing strict 2-D eruption evidence fields: {missing}")
    _validate_time_series(rows, label)
    gate = defs.result["hard_physics_gate"]["fixed_physical_event_test"]
    min_volume = float(gate["minimum_connected_water_volume_m3"])
    min_top = float(gate["minimum_connected_bulk_top_z_m"])
    rim = float(defs.result["hard_physics_gate"]["riser_rim_z_m"])
    raw_rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        sample_index = float(row["sample_index"])
        if not math.isfinite(sample_index) or not math.isclose(
            sample_index, round(sample_index), abs_tol=1e-10
        ):
            raise EvidenceError(f"{label}: sample_index is not an integer")
        target = float(row["target_time_s"])
        actual = float(row["actual_time_s"])
        elapsed = float(row["stage2_elapsed_s"])
        if not all(math.isfinite(value) for value in (target, actual, elapsed)):
            raise EvidenceError(f"{label}: event timing contains NaN/Inf")
        if ordinal == 0:
            stage2_start = target
        if not math.isclose(
            float(row["time_s"]), sample_index * defs.dt, rel_tol=0.0, abs_tol=1e-9
        ):
            raise EvidenceError(f"{label}: sample_index/time_s is not the frozen 0.10-s grid")
        if not math.isclose(target - stage2_start, float(row["time_s"]), abs_tol=1e-9):
            raise EvidenceError(f"{label}: target time is not the unshifted Stage-2 grid")
        if not math.isclose(actual - stage2_start, elapsed, abs_tol=1e-9):
            raise EvidenceError(f"{label}: stage2_elapsed_s conflicts with actual time")
        count = float(row["launch_component_count"])
        volume = float(row["component_water_volume_m3"])
        top = float(row["bulk_top_q99_z_m"])
        ratio = float(row["volume_over_minimum"])
        height = float(row["bulk_height_above_rim_m"])
        reported_raw = float(row["active_raw"])
        if not math.isfinite(count) or count < 0.0 or not math.isclose(count, round(count), abs_tol=1e-10):
            raise EvidenceError(f"{label}: launch_component_count is invalid")
        if not math.isfinite(volume) or volume < -1e-15:
            raise EvidenceError(f"{label}: connected water volume is invalid")
        if not math.isclose(ratio, volume / min_volume, rel_tol=1e-9, abs_tol=1e-12):
            raise EvidenceError(f"{label}: volume_over_minimum is inconsistent")
        if math.isfinite(top):
            if not math.isclose(height, top - rim, rel_tol=0.0, abs_tol=1e-9):
                raise EvidenceError(f"{label}: bulk height is inconsistent with q99 and rim")
        elif math.isfinite(height):
            raise EvidenceError(f"{label}: finite bulk height has no finite q99 top")
        computed_raw = bool(count >= 1.0 and volume >= min_volume - 1e-12 and top >= min_top - 1e-12)
        if reported_raw not in (0.0, 1.0) or bool(reported_raw) != computed_raw:
            raise EvidenceError(
                f"{label}: active_raw is not the frozen connected-volume/q99 predicate"
            )
        raw_rows.append(
            {
                "sample_index": int(round(sample_index)),
                "time_s": float(row["time_s"]),
                "active_raw": computed_raw,
            }
        )
    apply_persistence(raw_rows, defs)
    for source, expected in zip(rows, raw_rows, strict=True):
        reported_persistent = float(source["active_persistent"])
        canonical = float(source["internal_mouth_event_active"])
        expected_active = bool(expected["active_persistent"])
        if reported_persistent not in (0.0, 1.0) or bool(reported_persistent) != expected_active:
            raise EvidenceError(f"{label}: active_persistent is not the frozen 0.10-s result")
        if canonical not in (0.0, 1.0) or bool(canonical) != expected_active:
            raise EvidenceError(
                f"{label}: canonical eruption flag is not backed by strict spatial evidence"
            )
    return {
        "status": "strict_shared_face_connected_volume_q99_persistence_verified",
        "sample_count": len(rows),
        "raw_active_sample_count": sum(bool(row["active_raw"]) for row in raw_rows),
        "persistent_active_sample_count": sum(
            bool(row["active_persistent"]) for row in raw_rows
        ),
        "connectivity": "shared_cell_face",
        "bulk_top_quantile": float(gate["bulk_top_water_quantile"]),
        "minimum_continuous_duration_s": float(gate["minimum_continuous_duration_s"]),
    }


def validate_2d_unavailable_columns(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, float]],
    defs: Definitions,
    label: str,
) -> dict[str, Any]:
    unavailable = defs.data["observable_status"][
        "unavailable_from_alpha_water_and_U_only"
    ]
    names = [str(name) for name in unavailable]
    missing = sorted(set(names) - set(fields))
    if missing:
        raise EvidenceError(f"{label}: unavailable canonical columns are omitted: {missing}")
    invented = [
        name
        for name in names
        if any(math.isfinite(float(row.get(name, math.nan))) for row in rows)
    ]
    if invented:
        raise EvidenceError(
            f"{label}: alpha.water/U-only export invents unavailable observables: {invented}"
        )
    return {
        "status": "unavailable_columns_verified_empty",
        "columns": names,
        "reasons": unavailable,
    }


def common_grid(end_s: float, dt: float) -> np.ndarray:
    count = int(math.floor(end_s / dt + 1e-9)) + 1
    return np.arange(count, dtype=float) * dt


def resample_rows(
    rows: Sequence[Mapping[str, float]],
    columns: Sequence[str],
    targets: np.ndarray,
    *,
    max_gap_s: float,
    categorical: Iterable[str] = (),
) -> list[dict[str, float]]:
    times = _validate_time_series(rows, "input series")
    categorical_set = set(categorical)
    result = [{"time_s": float(target)} for target in targets]
    for name in columns:
        if name == "time_s":
            continue
        if name not in rows[0]:
            for target_row in result:
                target_row[name] = math.nan
            continue
        values = np.asarray([float(row.get(name, math.nan)) for row in rows], dtype=float)
        for target_row, target in zip(result, targets, strict=True):
            exact = np.flatnonzero(np.isclose(times, target, atol=1e-10, rtol=0.0))
            if len(exact):
                target_row[name] = float(values[int(exact[-1])])
                continue
            position = int(np.searchsorted(times, target, side="left"))
            if position == 0 or position == len(times):
                target_row[name] = math.nan
                continue
            left, right = position - 1, position
            if times[right] - times[left] > max_gap_s + 1e-12:
                target_row[name] = math.nan
                continue
            a, b = values[left], values[right]
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                target_row[name] = math.nan
                continue
            if name in categorical_set:
                # Zero-order hold; never create a fractional event state.
                target_row[name] = float(a)
            else:
                fraction = (target - times[left]) / (times[right] - times[left])
                target_row[name] = float(a + fraction * (b - a))
    return result


def _finite_pairs(
    one: Sequence[Mapping[str, float]],
    two: Sequence[Mapping[str, float]],
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(one) != len(two):
        raise EvidenceError("comparison series have different common-grid lengths")
    times: list[float] = []
    a: list[float] = []
    b: list[float] = []
    for left, right in zip(one, two, strict=True):
        if not math.isclose(float(left["time_s"]), float(right["time_s"]), abs_tol=1e-10):
            raise EvidenceError("comparison attempted a hidden time translation")
        av = float(left.get(name, math.nan))
        bv = float(right.get(name, math.nan))
        if math.isfinite(av) and math.isfinite(bv):
            times.append(float(left["time_s"]))
            a.append(av)
            b.append(bv)
    return np.asarray(times), np.asarray(a), np.asarray(b)


def waveform_metrics(
    one: Sequence[Mapping[str, float]],
    two: Sequence[Mapping[str, float]],
    name: str,
    minimum_points: int,
) -> dict[str, Any]:
    times, a, b = _finite_pairs(one, two, name)
    if len(a) < minimum_points:
        return {
            "status": "unavailable",
            "reason": f"only {len(a)} common finite samples",
            "sample_count": int(len(a)),
        }
    error = a - b
    scale = float(np.ptp(b))
    rmse = float(np.sqrt(np.mean(error**2)))
    peak_a_index = int(np.argmax(np.abs(a - np.mean(a))))
    peak_b_index = int(np.argmax(np.abs(b - np.mean(b))))
    correlation = (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 0.0 and np.std(b) > 0.0
        else None
    )
    return {
        "status": "compared_at_zero_time_shift",
        "sample_count": int(len(a)),
        "start_s": float(times[0]),
        "end_s": float(times[-1]),
        "signed_mean_error_1d_minus_2d": float(np.mean(error)),
        "rmse": rmse,
        "normalized_rmse_by_2d_range": rmse / scale if scale > 0.0 else None,
        "max_absolute_error": float(np.max(np.abs(error))),
        "correlation_at_zero_lag": correlation,
        "signature_peak_amplitude_error_1d_minus_2d": float(a[peak_a_index] - b[peak_b_index]),
        "phase_peak_time_error_s_1d_minus_2d": float(times[peak_a_index] - times[peak_b_index]),
        "time_shift_applied_s": 0.0,
    }


def _boolean_events(
    rows: Sequence[Mapping[str, float]], name: str, dt: float
) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        value = float(row.get(name, math.nan))
        active = math.isfinite(value) and value >= 0.5
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(rows) - 1):
            end = index if active and index == len(rows) - 1 else index - 1
            events.append((float(rows[start]["time_s"]), float(rows[end]["time_s"])))
            start = None
    return events


def branch_status(
    rows: Sequence[Mapping[str, float]], defs: Definitions
) -> dict[str, Any]:
    event_values = [
        float(row.get("internal_mouth_event_active", math.nan)) for row in rows
    ]
    candidate_events = _boolean_events(rows, "internal_mouth_event_active", defs.dt)
    minimum_duration = float(
        defs.result["hard_physics_gate"]["fixed_physical_event_test"]
        ["minimum_continuous_duration_s"]
    )
    events = [
        (start, end)
        for start, end in candidate_events
        if end - start >= minimum_duration - 1e-9
    ]
    end = float(rows[-1]["time_s"])
    planned = float(defs.data["comparison"]["no_event_final_decision_time_s"])
    if events:
        decision: bool | None = True
        status = "eruption"
    elif any(not math.isfinite(value) for value in event_values):
        decision = None
        status = "inconclusive_missing_event_samples"
    elif end + 1e-9 < planned:
        decision = None
        status = "inconclusive_no_event_before_planned_end"
    else:
        decision = False
        status = "complete_no_eruption"
    return {
        "status": status,
        "eruption_decision": decision,
        "series_end_s": end,
        "event_count": len(events),
        "discarded_short_event_count": len(candidate_events) - len(events),
        "minimum_event_duration_s": minimum_duration,
        "finite_event_sample_count": sum(math.isfinite(value) for value in event_values),
        "events": [
            {"onset_s": a, "end_s": b, "duration_s": b - a} for a, b in events
        ],
    }


def _post_event_period_proxy(
    rows: Sequence[Mapping[str, float]],
    event_end_s: float | None,
    defs: Definitions,
) -> float | None:
    if event_end_s is None:
        return None
    config = defs.data["comparison"]["post_eruption_period_proxy"]
    name = str(config["pressure_series"])
    end_s = event_end_s + float(config["window_after_first_event_end_s"])
    selected = [
        (float(row["time_s"]), float(row.get(name, math.nan)))
        for row in rows
        if event_end_s - 1e-9 <= float(row["time_s"]) <= end_s + 1e-9
        and math.isfinite(float(row.get(name, math.nan)))
    ]
    if len(selected) < 5:
        return None
    times = np.asarray([item[0] for item in selected])
    values = np.asarray([item[1] for item in selected])
    if np.max(np.abs(np.diff(times) - defs.dt)) > 1e-8:
        return None
    # Remove a least-squares line so recovery drift is not mistaken for a period.
    design = np.column_stack((np.ones(len(times)), times - times[0]))
    trend = design @ np.linalg.lstsq(design, values, rcond=None)[0]
    signal = values - trend
    variance = float(np.dot(signal, signal))
    if variance <= 0.0:
        return None
    low, high = (float(item) for item in config["search_period_s"])
    min_lag = max(1, int(math.ceil(low / defs.dt - 1e-9)))
    max_lag = min(len(signal) - 2, int(math.floor(high / defs.dt + 1e-9)))
    if max_lag < min_lag:
        return None
    correlations = {
        lag: float(np.dot(signal[:-lag], signal[lag:]))
        / math.sqrt(float(np.dot(signal[:-lag], signal[:-lag]) * np.dot(signal[lag:], signal[lag:])))
        for lag in range(min_lag, max_lag + 1)
        if np.dot(signal[:-lag], signal[:-lag]) > 0.0
        and np.dot(signal[lag:], signal[lag:]) > 0.0
    }
    local = [
        lag
        for lag in range(min_lag + 1, max_lag)
        if lag in correlations
        and lag - 1 in correlations
        and lag + 1 in correlations
        and correlations[lag] >= correlations[lag - 1]
        and correlations[lag] >= correlations[lag + 1]
    ]
    if not local:
        return None
    best = max(local, key=lambda lag: correlations[lag])
    return best * defs.dt


def _signature(
    rows: Sequence[Mapping[str, float]], defs: Definitions, *, dimensionality: str
) -> dict[str, Any]:
    branch = branch_status(rows, defs)
    event_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("internal_mouth_event_active", math.nan)))
        and float(row["internal_mouth_event_active"]) >= 0.5
    ]
    rim = float(defs.data["geometry"]["riser_rim_z_m"])
    external_q99_heights = [
        float(row.get("bulk_top_q99_z_m", math.nan)) - rim
        for row in rows
        if dimensionality == "2d"
        and math.isfinite(float(row.get("active_persistent", math.nan)))
        and float(row["active_persistent"]) >= 0.5
        and math.isfinite(float(row.get("bulk_top_q99_z_m", math.nan)))
    ]
    speeds = [
        float(row.get("horizontal_slug_velocity_m_s", math.nan))
        for row in rows
        if math.isfinite(float(row.get("horizontal_slug_velocity_m_s", math.nan)))
    ]
    arrivals = [
        float(row["time_s"])
        for row in rows
        if math.isfinite(float(row.get("gas_arrival_at_riser", math.nan)))
        and float(row["gas_arrival_at_riser"]) >= 0.5
    ]
    onsets = [float(item["onset_s"]) for item in branch["events"]]
    cycle = onsets[1] - onsets[0] if len(onsets) >= 2 else None
    first_end = float(branch["events"][0]["end_s"]) if branch["events"] else None
    internal_tops = [
        float(row.get("riser_connected_water_top_z_m", math.nan))
        for row in rows
        if math.isfinite(float(row.get("riser_connected_water_top_z_m", math.nan)))
    ]
    external_height = max(external_q99_heights) if external_q99_heights else None
    return {
        "branch": branch,
        "first_eruption_onset_s": onsets[0] if onsets else None,
        "first_eruption_duration_s": (
            float(branch["events"][0]["duration_s"]) if branch["events"] else None
        ),
        "cycle_between_first_two_onsets_s": cycle,
        "gas_arrival_at_riser_s": arrivals[0] if arrivals else None,
        "maximum_internal_or_connected_water_top_z_m": max(internal_tops) if internal_tops else None,
        "maximum_external_launch_connected_bulk_q99_height_above_rim_m": external_height,
        "external_height_status": (
            "available_from_2d_external_cells"
            if dimensionality == "2d"
            else "unavailable_without_resolved_1d_external_free_surface_domain"
        ),
        "fig8_middle_slug_unmixed_water_speed_mean_m_s_proxy": (
            float(np.mean(speeds)) if speeds else None
        ),
        "fig8_middle_slug_unmixed_water_speed_max_m_s_proxy": max(speeds) if speeds else None,
        "fig8_semantics": "water_velocity_magnitude_not_gas_or_interface_speed",
        "post_eruption_P2_oscillation_period_s_proxy": _post_event_period_proxy(
            rows, first_end, defs
        ),
        "event_sample_count": len(event_rows),
    }


def signature_error(one: Mapping[str, Any], two: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "first_eruption_onset_s",
        "first_eruption_duration_s",
        "cycle_between_first_two_onsets_s",
        "gas_arrival_at_riser_s",
        "maximum_external_launch_connected_bulk_q99_height_above_rim_m",
        "fig8_middle_slug_unmixed_water_speed_mean_m_s_proxy",
        "fig8_middle_slug_unmixed_water_speed_max_m_s_proxy",
        "post_eruption_P2_oscillation_period_s_proxy",
    ]
    output: dict[str, Any] = {}
    for name in fields:
        a = one.get(name)
        b = two.get(name)
        if a is None or b is None:
            output[name] = {"status": "unavailable"}
            continue
        signed = float(a) - float(b)
        output[name] = {
            "status": "compared_at_zero_time_shift",
            "one_d": float(a),
            "two_d": float(b),
            "signed_error_1d_minus_2d": signed,
            "relative_error_to_2d": signed / float(b) if float(b) != 0.0 else None,
            "time_shift_applied_s": 0.0,
        }
    return output


def _point_target(value: Any, target: float) -> dict[str, Any]:
    if value is None:
        return {"status": "unavailable", "target": target}
    signed = float(value) - target
    return {
        "status": "reported_no_published_pass_tolerance",
        "value": float(value),
        "target": target,
        "signed_error_value_minus_target": signed,
        "relative_error_to_target": signed / target if target != 0.0 else None,
    }


def _range_target(value: Any, bounds: Sequence[float]) -> dict[str, Any]:
    low, high = (float(item) for item in bounds)
    if value is None:
        return {"status": "unavailable", "target_range": [low, high]}
    number = float(value)
    nearest_error = number - min(max(number, low), high)
    return {
        "status": "reported_no_published_pass_tolerance",
        "value": number,
        "target_range": [low, high],
        "within_target_range": low <= number <= high,
        "signed_error_to_nearest_range_edge": nearest_error,
    }


def source_target_report(
    signature: Mapping[str, Any], defs: Definitions, *, dimensionality: str
) -> dict[str, Any]:
    targets = defs.result["published_and_figure_read_targets"]
    report = {
        "eruption": {
            "target": bool(targets["eruption"]["target"]),
            "value": signature["branch"]["eruption_decision"],
            "status": (
                "matched"
                if signature["branch"]["eruption_decision"]
                == bool(targets["eruption"]["target"])
                else (
                    "inconclusive"
                    if signature["branch"]["eruption_decision"] is None
                    else "mismatched"
                )
            ),
        },
        "eruption_duration_vs_experiment_s": _point_target(
            signature.get("first_eruption_duration_s"),
            float(targets["eruption_duration_s"]["experiment"]),
        ),
        "complete_cycle_s": _point_target(
            signature.get("cycle_between_first_two_onsets_s"),
            float(targets["complete_cycle_s"]["target"]),
        ),
        "fig8_unmixed_middle_slug_water_speed_m_s": _range_target(
            signature.get("fig8_middle_slug_unmixed_water_speed_mean_m_s_proxy"),
            targets["horizontal_slug_velocity_m_per_s"]["target"],
        ),
        "post_eruption_P2_period_s_proxy": _point_target(
            signature.get("post_eruption_P2_oscillation_period_s_proxy"),
            float(targets["post_eruption_oscillation_period_s"]["target"]),
        ),
    }
    if dimensionality == "2d":
        report["maximum_height_above_rim_vs_experiment_m"] = _point_target(
            signature.get("maximum_external_launch_connected_bulk_q99_height_above_rim_m"),
            float(targets["maximum_height_above_rim_m"]["experiment"]),
        )
    else:
        report["maximum_height_above_rim_vs_experiment_m"] = {
            "status": "unavailable_without_resolved_external_free_surface_domain",
            "target": float(targets["maximum_height_above_rim_m"]["experiment"]),
            "forbidden_substitution": "internal_riser_water_top",
            "allowed_separate_label": "derived_plume_proxy",
            "note": (
                "A persistent reduced-order exterior plume inventory is not a resolved "
                "external free-surface height field."
            ),
        }
    return report


def validate_profile_npz(path: Path, defs: Definitions) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"missing 1-D profile NPZ: {path}")
    required = [str(item) for item in defs.common["required_profile_fields"]]
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise EvidenceError(f"1-D profile NPZ omits fields: {missing}")
        times = np.asarray(archive["time_s"], dtype=float)
        z = np.asarray(archive["riser_z_cell_center_m"], dtype=float)
        if times.ndim != 1 or z.ndim not in (1, 2):
            raise EvidenceError("profile time and z arrays have inadmissible dimensions")
        if len(times) == 0 or not math.isclose(float(times[0]), 0.0, abs_tol=1e-10):
            raise EvidenceError("1-D profile must begin at unshifted Stage-2 t=0")
        if np.any(np.diff(times) <= 0.0) or np.any(~np.isfinite(times)):
            raise EvidenceError("profile times must be finite and strictly increasing")
        max_time_grid_error = float(np.max(np.abs(times / defs.dt - np.rint(times / defs.dt))))
        if max_time_grid_error > 1e-8:
            raise EvidenceError("1-D profile times do not lie on the frozen common grid")
        shapes: dict[str, list[int]] = {}
        expected_profile_shape = (
            (len(times), len(z)) if z.ndim == 1 else tuple(int(item) for item in z.shape)
        )
        if z.ndim == 2 and z.shape[0] != len(times):
            raise EvidenceError("time-dependent riser z grid first axis is not time")
        for name in required:
            values = np.asarray(archive[name])
            if np.any(np.isinf(values)):
                raise EvidenceError(f"profile {name} contains Inf")
            shapes[name] = list(values.shape)
            if name not in ("time_s", "riser_z_cell_center_m") and values.shape[0] != len(times):
                raise EvidenceError(f"profile {name} first axis is not time")
            if name not in ("time_s", "riser_z_cell_center_m") and values.shape != expected_profile_shape:
                raise EvidenceError(
                    f"profile {name} shape {values.shape} != riser grid {expected_profile_shape}"
                )
        qup = np.asarray(archive["riser_Qup_m3_s"], dtype=float)
        qdown = np.asarray(archive["riser_Qdown_m3_s"], dtype=float)
        aup = np.asarray(archive["riser_Aup_m2"], dtype=float)
        adown = np.asarray(archive["riser_Adown_m2"], dtype=float)
        if np.any(qup < -1e-12) or np.any(qdown < -1e-12):
            raise EvidenceError("Qup/Qdown must be separately non-negative gross streams")
        if np.any(aup < -1e-12) or np.any(adown < -1e-12):
            raise EvidenceError("Aup/Adown must be non-negative")
        separately_stored = "riser_Qup_m3_s" in archive.files and "riser_Qdown_m3_s" in archive.files
        values_not_identical = not np.array_equal(qup, qdown)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "required_fields": required,
        "shapes": shapes,
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "common_grid_s": defs.dt,
        "time_shift_applied_s": 0.0,
        "Qup_Qdown_stored_as_separate_keys": separately_stored,
        "Qup_Qdown_values_not_identical": values_not_identical,
        "note": (
            "Qup/Qdown are required as separate keys. Equal values would not prove "
            "reconstruction when a trajectory is quiescent; the comparison never derives "
            "either gross stream from net flow."
        ),
    }


def profile_section_consistency(
    path: Path,
    canonical_rows: Sequence[Mapping[str, float]],
    defs: Definitions,
) -> dict[str, Any]:
    target_z = float(defs.data["field_extraction"]["riser_flux_section_z_m"])
    profile_rows: list[dict[str, float]] = []
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["time_s"], dtype=float)
        z_all = np.asarray(archive["riser_z_cell_center_m"], dtype=float)
        qup = np.asarray(archive["riser_Qup_m3_s"], dtype=float)
        qdown = np.asarray(archive["riser_Qdown_m3_s"], dtype=float)
        for index, time_s in enumerate(times):
            z = z_all if z_all.ndim == 1 else z_all[index]
            order = np.argsort(z)
            z = z[order]
            row: dict[str, float] = {"time_s": float(time_s)}
            for source, output in (
                (qup[index][order], "riser_upward_liquid_flow_m3_s"),
                (qdown[index][order], "riser_downward_liquid_flow_m3_s"),
            ):
                finite = np.isfinite(z) & np.isfinite(source)
                if (
                    np.count_nonzero(finite) >= 2
                    and target_z >= float(np.min(z[finite])) - 1e-12
                    and target_z <= float(np.max(z[finite])) + 1e-12
                ):
                    row[output] = float(np.interp(target_z, z[finite], source[finite]))
                else:
                    row[output] = math.nan
            profile_rows.append(row)
    minimum = int(defs.data["comparison"]["minimum_points_for_waveform_metric"])

    def consistency(name: str) -> dict[str, Any]:
        _, profile_values, canonical_values = _finite_pairs(
            profile_rows, canonical_rows, name
        )
        if len(profile_values) < minimum:
            return {
                "status": "unavailable",
                "sample_count": int(len(profile_values)),
            }
        error = profile_values - canonical_values
        return {
            "status": "checked_at_identical_unshifted_times",
            "sample_count": int(len(error)),
            "signed_mean_profile_minus_canonical": float(np.mean(error)),
            "rmse_profile_vs_canonical": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_difference": float(np.max(np.abs(error))),
        }

    return {
        "section_z_m": target_z,
        "spatial_mapping": "linear_interpolation_on_native_riser_cell_centres",
        "time_shift_applied_s": 0.0,
        "Qup_profile_vs_canonical_scalar": consistency(
            "riser_upward_liquid_flow_m3_s"
        ),
        "Qdown_profile_vs_canonical_scalar": consistency(
            "riser_downward_liquid_flow_m3_s"
        ),
    }


def grid_spread(
    meshes: Mapping[str, Sequence[Mapping[str, float]]],
    name: str,
    minimum_points: int,
) -> dict[str, Any]:
    levels = list(meshes)
    if len(levels) < 2:
        return {"status": "unavailable", "reason": "fewer than two 2-D meshes"}
    count = min(len(meshes[level]) for level in levels)
    values: list[list[float]] = []
    times: list[float] = []
    for index in range(count):
        row_values: list[float] = []
        reference_time = float(meshes[levels[0]][index]["time_s"])
        aligned = True
        for level in levels:
            row = meshes[level][index]
            if not math.isclose(float(row["time_s"]), reference_time, abs_tol=1e-10):
                aligned = False
                break
            value = float(row.get(name, math.nan))
            if not math.isfinite(value):
                aligned = False
                break
            row_values.append(value)
        if aligned:
            times.append(reference_time)
            values.append(row_values)
    if len(values) < minimum_points:
        return {
            "status": "unavailable",
            "reason": f"only {len(values)} all-mesh finite samples",
        }
    matrix = np.asarray(values, dtype=float)
    span = np.max(matrix, axis=1) - np.min(matrix, axis=1)
    pairwise: dict[str, float] = {}
    for left_index, left in enumerate(levels):
        for right_index in range(left_index + 1, len(levels)):
            right = levels[right_index]
            difference = matrix[:, left_index] - matrix[:, right_index]
            pairwise[f"{left}_minus_{right}_rmse"] = float(
                np.sqrt(np.mean(difference**2))
            )
    return {
        "status": "reported_separately_from_1d_error",
        "sample_count": len(values),
        "start_s": times[0],
        "end_s": times[-1],
        "mean_pointwise_max_minus_min": float(np.mean(span)),
        "maximum_pointwise_max_minus_min": float(np.max(span)),
        "rms_pointwise_max_minus_min": float(np.sqrt(np.mean(span**2))),
        "pairwise_rmse": pairwise,
    }


def compare_1d_to_2d(
    *,
    one_d_csv: Path,
    one_d_profile_npz: Path,
    mesh_csvs: Mapping[str, Path],
    defs: Definitions,
) -> dict[str, Any]:
    required_levels = [str(item) for item in defs.data["comparison"]["required_mesh_levels"]]
    missing_levels = [level for level in required_levels if level not in mesh_csvs]
    extra_levels = [level for level in mesh_csvs if level not in required_levels]
    if missing_levels or extra_levels:
        raise EvidenceError(
            f"mesh inputs must be exactly {required_levels}; missing={missing_levels}, extra={extra_levels}"
        )
    one_fields, one_native = read_numeric_csv(one_d_csv)
    missing_canonical = [name for name in defs.canonical if name not in one_fields]
    if missing_canonical:
        raise EvidenceError(f"1-D canonical CSV omits fields: {missing_canonical}")
    one_times = _validate_time_series(one_native, "1-D canonical CSV")

    mesh_native: dict[str, list[dict[str, float]]] = {}
    mesh_hashes: dict[str, str] = {}
    mesh_event_evidence: dict[str, dict[str, Any]] = {}
    mesh_unavailable_evidence: dict[str, dict[str, Any]] = {}
    mesh_provenance: dict[str, dict[str, Any]] = {}
    for level in required_levels:
        fields, rows = read_numeric_csv(mesh_csvs[level])
        mesh_provenance[level] = validate_2d_metadata(mesh_csvs[level], level, defs)
        missing = [name for name in defs.canonical if name not in fields]
        if missing:
            raise EvidenceError(f"2-D {level} CSV omits canonical fields: {missing}")
        _validate_time_series(rows, f"2-D {level} CSV")
        mesh_event_evidence[level] = validate_2d_event_evidence(
            fields, rows, defs, f"2-D {level} CSV"
        )
        mesh_unavailable_evidence[level] = validate_2d_unavailable_columns(
            fields, rows, defs, f"2-D {level} CSV"
        )
        mesh_native[level] = rows
        mesh_hashes[level] = sha256_file(mesh_csvs[level])
    if len(set(mesh_hashes.values())) != len(required_levels):
        raise EvidenceError(
            "the three mesh labels do not identify three distinct 2-D evidence files"
        )

    common_end = min(
        float(one_times[-1]),
        *(float(rows[-1]["time_s"]) for rows in mesh_native.values()),
    )
    targets = common_grid(common_end, defs.dt)
    categorical = {"gas_arrival_at_riser", "internal_mouth_event_active"}
    # Native 1-D output may be finer than 0.10 s; only local bracketing is allowed.
    max_gap = defs.dt + 1e-12
    one = resample_rows(
        one_native,
        defs.canonical,
        targets,
        max_gap_s=max_gap,
        categorical=categorical,
    )
    strict_event_fields = [
        str(item)
        for item in defs.result["required_acceptance_outputs"]["time_series_fields"]
    ]
    mesh_columns = list(dict.fromkeys([*defs.canonical, *strict_event_fields]))
    meshes = {
        level: resample_rows(
            rows,
            mesh_columns,
            targets,
            max_gap_s=max_gap,
            categorical={*categorical, "active_raw", "active_persistent"},
        )
        for level, rows in mesh_native.items()
    }

    minimum = int(defs.data["comparison"]["minimum_points_for_waveform_metric"])
    scalar_names = [str(item) for item in defs.data["comparison"]["scalar_series"]]
    per_mesh: dict[str, Any] = {}
    one_signature = _signature(one, defs, dimensionality="1d")
    one_branch = one_signature["branch"]
    for level in required_levels:
        mesh_signature = _signature(meshes[level], defs, dimensionality="2d")
        per_mesh[level] = {
            "two_d_branch": mesh_signature["branch"],
            "one_d_vs_two_d_signature_errors": signature_error(
                one_signature, mesh_signature
            ),
            "waveform_metrics": {
                name: waveform_metrics(one, meshes[level], name, minimum)
                for name in scalar_names
            },
            "two_d_signature": mesh_signature,
            "two_d_vs_published_targets": source_target_report(
                mesh_signature, defs, dimensionality="2d"
            ),
        }

    expected = bool(defs.result["hard_physics_gate"]["expected_eruption"])
    decisions: dict[str, bool | None] = {
        "one_d": one_branch["eruption_decision"],
        **{
            level: per_mesh[level]["two_d_branch"]["eruption_decision"]
            for level in required_levels
        },
    }
    if any(value is None for value in decisions.values()):
        hard_status = "INCONCLUSIVE_INCOMPLETE_BRANCH_EVIDENCE"
    elif all(value == expected for value in decisions.values()):
        hard_status = "PASS_ERUPTION_BRANCH_MATCH"
    else:
        hard_status = "FAIL_PHYSICS_ALIGNMENT_ERUPTION_BRANCH"

    profile = validate_profile_npz(one_d_profile_npz, defs)
    profile_consistency = profile_section_consistency(
        one_d_profile_npz, one, defs
    )
    proxy_names = set(defs.data["observable_status"]["declared_cell_center_proxies"])
    return {
        "schema_version": 1,
        "case_id": defs.data["case_id"],
        "comparison_identity": {
            "physical_condition_count": 1,
            "two_d_mesh_levels": required_levels,
            "note": "The three meshes are not three experimental conditions.",
        },
        "time_alignment": {
            "origin": "stage_2_air_opening",
            "common_grid_s": defs.dt,
            "time_shift_allowed": False,
            "time_shift_applied_s": 0.0,
            "comparison_end_s": float(targets[-1]),
        },
        "hard_eruption_gate": {
            "published_expected_eruption": expected,
            "decisions": decisions,
            "status": hard_status,
            "rule": defs.data["comparison"]["categorical_gate"],
        },
        "one_d_signature": one_signature,
        "one_d_vs_published_targets": source_target_report(
            one_signature, defs, dimensionality="1d"
        ),
        "per_mesh": per_mesh,
        "mesh_spread_separate_from_1d_error": {
            name: grid_spread(meshes, name, minimum) for name in scalar_names
        },
        "observable_semantics": {
            "fig8_velocity": defs.data["source_semantics"]["fig8_velocity"],
            "two_d_proxies": sorted(proxy_names),
            "two_d_unavailable": defs.data["observable_status"]
            ["unavailable_from_alpha_water_and_U_only"],
        },
        "two_d_strict_event_evidence": mesh_event_evidence,
        "two_d_unavailable_evidence": mesh_unavailable_evidence,
        "two_d_mesh_provenance": mesh_provenance,
        "one_d_profile_validation": profile,
        "one_d_profile_to_canonical_consistency": profile_consistency,
        "profile_cross_dimensional_scope": defs.data["observable_status"]
        ["profile_comparability"],
        "input_evidence": {
            "one_d_csv": str(one_d_csv.resolve()),
            "one_d_csv_sha256": sha256_file(one_d_csv),
            "one_d_profile_npz": str(one_d_profile_npz.resolve()),
            "mesh_csv_sha256": mesh_hashes,
            "definition_sha256": sha256_file(defs.path),
            "result_acceptance_sha256": sha256_file(defs.result_path),
            "common_observables_sha256": sha256_file(defs.common_path),
        },
        "result_marker_written": False,
        "acceptance_scope": (
            "Diagnostic comparison only. This JSON cannot create or imply RESULT_ACCEPTED."
        ),
    }
