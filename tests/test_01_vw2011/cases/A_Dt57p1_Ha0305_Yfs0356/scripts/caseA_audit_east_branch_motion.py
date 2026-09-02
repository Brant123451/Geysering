"""Audit the Case-A east-branch gas motion against archived 2-D VOF fields.

The 2-D crown front is a diagnostic contour, not a boundary condition for the
1-D model.  A column belongs to the crown-gas envelope when its uppermost pipe
cell has ``alpha.water < 0.9``.  The integrated comparison uses the unsmoothed,
area-weighted column liquid fraction.  The 1-D envelope uses the archived
material liquid fraction and a 5% void threshold.  All definitions and their
spatial/temporal resolution are written to the output JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
VTK_ROOT = CASE / "openfoam/2d/VTK_CASEA_HTML"

D = 0.094
L = 4.006
TOWER_X = 3.516
TOWER_DIAMETER = 0.0571
EAST_START = TOWER_X + 0.5 * TOWER_DIAMETER
LOCAL_X = 3.779
TOP_ALPHA_THRESHOLD = 0.90
ONE_D_VOID_THRESHOLD = 0.05


def _array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Expected ASCII foamToVTK output")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def _read_field(path: Path, *, geometry: bool) -> dict[str, object]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    if time_node is None or alpha_node is None:
        raise ValueError(f"Missing TimeValue or alpha.water in {path}")
    result: dict[str, object] = {
        "time": float((time_node.text or "0").strip()),
        "alpha": np.clip(_array(alpha_node), 0.0, 1.0),
    }
    if not geometry:
        return result

    points_node = root.find(".//Points/DataArray")
    if points_node is None:
        raise ValueError(f"Missing points in {path}")
    points = _array(points_node, 3)
    arrays = {
        node.attrib.get("Name"): node
        for node in root.findall(".//Cells/DataArray")
    }
    connectivity = _array(arrays["connectivity"]).astype(int)
    offsets = _array(arrays["offsets"]).astype(int)
    centres = []
    areas = []
    start = 0
    for stop in offsets:
        cell = connectivity[start:stop]
        start = int(stop)
        xy = np.unique(points[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(
            np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
        )
        polygon = xy[order]
        x = polygon[:, 0]
        y = polygon[:, 1]
        centres.append(centre)
        areas.append(
            0.5
            * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        )
    result.update(
        centres=np.asarray(centres, dtype=float),
        areas=np.asarray(areas, dtype=float),
    )
    return result


def _paths() -> list[tuple[float, Path]]:
    found = []
    for path in VTK_ROOT.glob("*/internal.vtu"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.read(512)
        match = re.search(r"\btime='([^']+)'", header)
        if match is None:
            raise ValueError(f"Cannot read time from {path}")
        time = float(match.group(1))
        if -1.0e-8 <= time <= 13.0 + 1.0e-8:
            found.append((time, path))
    found.sort(key=lambda item: item[0])
    if len(found) != 261:
        raise RuntimeError(f"Expected 261 2-D frames, found {len(found)}")
    return found


def _first_crossing(time: np.ndarray, position: np.ndarray, x: float) -> float | None:
    indices = np.flatnonzero(np.isfinite(position) & (position >= x))
    return None if indices.size == 0 else float(time[indices[0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-d-fields", type=Path, required=True)
    parser.add_argument("--one-d-diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields_path = args.one_d_fields
    diagnostics_path = args.one_d_diagnostics
    output_path = args.output
    if not fields_path.is_absolute():
        fields_path = CASE / fields_path
    if not diagnostics_path.is_absolute():
        diagnostics_path = CASE / diagnostics_path
    if not output_path.is_absolute():
        output_path = CASE / output_path

    paths = _paths()
    first = _read_field(paths[0][1], geometry=True)
    centres = np.asarray(first["centres"], dtype=float)
    areas = np.asarray(first["areas"], dtype=float)
    pipe = (
        (centres[:, 0] >= -1.0e-8)
        & (centres[:, 0] <= L + 1.0e-8)
        & (centres[:, 1] >= -0.5 * D - 1.0e-8)
        & (centres[:, 1] <= 0.5 * D + 1.0e-8)
    )
    x2, column = np.unique(np.round(centres[pipe, 0], 8), return_inverse=True)
    column_area = np.bincount(column, weights=areas[pipe])
    top_cell = np.empty(x2.size, dtype=int)
    pipe_indices = np.flatnonzero(pipe)
    for index in range(x2.size):
        candidates = pipe_indices[column == index]
        top_cell[index] = int(candidates[np.argmax(centres[candidates, 1])])
    east2 = x2 >= EAST_START
    dx2 = float(np.median(np.diff(x2)))
    local2 = int(np.argmin(np.abs(x2 - LOCAL_X)))

    time2 = []
    rear2 = []
    front2 = []
    alpha_local2 = []
    length2 = []
    for frame_index, (time_hint, path) in enumerate(paths):
        field = first if frame_index == 0 else _read_field(path, geometry=False)
        alpha = np.asarray(field["alpha"], dtype=float)
        time = float(field["time"])
        if abs(time - time_hint) > 1.0e-5:
            raise ValueError(f"VTK time mismatch in {path}")
        column_alpha = np.bincount(
            column, weights=alpha[pipe] * areas[pipe]
        ) / column_area
        crown = east2 & (alpha[top_cell] < TOP_ALPHA_THRESHOLD)
        positions = x2[crown]
        time2.append(time)
        rear2.append(float(positions[0]) if positions.size else np.nan)
        front2.append(float(positions[-1]) if positions.size else np.nan)
        alpha_local2.append(float(column_alpha[local2]))
        length2.append(float(np.sum((1.0 - column_alpha[east2]) * dx2)))

    saved = np.load(fields_path)
    time1 = np.asarray(saved["time"], dtype=float)
    alpha1 = np.asarray(saved["horizontal_alpha_l"], dtype=float)
    n1 = alpha1.shape[1]
    dx1 = L / n1
    x1 = (np.arange(n1, dtype=float) + 0.5) * dx1
    east1 = x1 >= EAST_START
    local1 = int(np.argmin(np.abs(x1 - LOCAL_X)))
    rear1 = np.full(time1.size, np.nan)
    front1 = np.full(time1.size, np.nan)
    for index, row in enumerate(alpha1):
        positions = x1[east1 & (row <= 1.0 - ONE_D_VOID_THRESHOLD)]
        if positions.size:
            rear1[index] = positions[0]
            front1[index] = positions[-1]
    length1 = np.sum((1.0 - alpha1[:, east1]) * dx1, axis=1)
    local_alpha1 = alpha1[:, local1]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostic_time = np.asarray(diagnostics["t"], dtype=float)
    material_front1 = np.interp(
        time1,
        diagnostic_time,
        np.asarray(diagnostics["side_t_east_material_front"], dtype=float),
    )

    time2_array = np.asarray(time2, dtype=float)
    front2_array = np.asarray(front2, dtype=float)
    length2_array = np.asarray(length2, dtype=float)
    front1_on_2 = np.interp(time2_array, time1, material_front1)
    length1_on_2 = np.interp(time2_array, time1, length1)
    valid_front = np.isfinite(front2_array)
    front_rmse = float(np.sqrt(np.mean(
        (front1_on_2[valid_front] - front2_array[valid_front]) ** 2
    ))) if np.any(valid_front) else None
    valid_east_time = time2_array >= 8.5
    length_rmse = float(np.sqrt(np.mean(
        (length1_on_2[valid_east_time] - length2_array[valid_east_time]) ** 2
    )))

    sample_times = np.asarray([8.5, 9.0, 9.5, 10.0, 11.0, 12.0, 12.5, 13.0])
    samples = []
    for time in sample_times:
        i2 = int(np.argmin(np.abs(time2_array - time)))
        i1 = int(np.argmin(np.abs(time1 - time)))
        samples.append({
            "time_s": float(time2_array[i2]),
            "two_d_crown_rear_m": None if not np.isfinite(rear2[i2]) else rear2[i2],
            "two_d_crown_front_m": None if not np.isfinite(front2_array[i2]) else float(front2_array[i2]),
            "one_d_void_rear_m": None if not np.isfinite(rear1[i1]) else float(rear1[i1]),
            "one_d_void_front_m": None if not np.isfinite(front1[i1]) else float(front1[i1]),
            "one_d_material_front_m": float(material_front1[i1]),
            "two_d_local_alpha_l": float(alpha_local2[i2]),
            "one_d_local_alpha_l": float(local_alpha1[i1]),
            "two_d_equivalent_gas_length_m": float(length2_array[i2]),
            "one_d_equivalent_gas_length_m": float(length1[i1]),
        })

    output = {
        "sources": {
            "two_d_vtk_root": str(VTK_ROOT),
            "one_d_fields": str(fields_path),
            "one_d_diagnostics": str(diagnostics_path),
        },
        "definitions": {
            "east_branch_start_m": EAST_START,
            "two_d_crown_envelope": (
                f"uppermost pipe cell alpha.water < {TOP_ALPHA_THRESHOLD}"
            ),
            "two_d_inventory": (
                "unsmoothed area-weighted column integral of 1-alpha.water"
            ),
            "one_d_envelope": (
                f"archived material alpha_l <= {1.0 - ONE_D_VOID_THRESHOLD}"
            ),
            "one_d_inventory": "integral of archived material 1-alpha_l",
            "local_probe_x_m": LOCAL_X,
            "two_d_dx_m": dx2,
            "one_d_dx_m": dx1,
            "output_interval_s": 0.05,
            "note": (
                "The 2-D diagnostic is an audit reference only; no extracted "
                "position, time, or liquid fraction is imposed on the 1-D solve."
            ),
        },
        "summary": {
            "two_d_first_crown_gas_time_s": (
                None
                if not np.any(np.isfinite(front2_array))
                else float(time2_array[np.flatnonzero(np.isfinite(front2_array))[0]])
            ),
            "one_d_first_east_void_time_s": (
                None
                if not np.any(np.isfinite(front1))
                else float(time1[np.flatnonzero(np.isfinite(front1))[0]])
            ),
            "front_rmse_over_two_d_detected_window_m": front_rmse,
            "equivalent_gas_length_rmse_8p5_to_13s_m": length_rmse,
            "arrival_times_s": {
                str(position): {
                    "two_d": _first_crossing(time2_array, front2_array, position),
                    "one_d": _first_crossing(time1, material_front1, position),
                }
                for position in (3.60, 3.65, 3.70, 3.75, 3.776, 3.80)
            },
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
