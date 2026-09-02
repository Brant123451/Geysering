#!/usr/bin/env python3
"""Extract a base-connected riser water column from planar VTP snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def _data_array(root: ET.Element, name: str) -> ET.Element:
    for element in root.iter("DataArray"):
        if element.attrib.get("Name") == name:
            return element
    raise KeyError(f"VTP DataArray {name!r} not found")


def _read_vtp(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    root = ET.fromstring(payload)
    time = float(np.fromstring(_data_array(root, "TimeValue").text or "", sep=" ")[0])
    points = np.fromstring(
        _data_array(root, "Points").text or "", sep=" ", dtype=float
    ).reshape(-1, 3)
    connectivity = np.fromstring(
        _data_array(root, "connectivity").text or "", sep=" ", dtype=np.int64
    )
    offsets = np.fromstring(
        _data_array(root, "offsets").text or "", sep=" ", dtype=np.int64
    )
    widths = np.diff(np.r_[0, offsets])
    if not np.all(widths == 3):
        raise ValueError(f"{path}: only triangular VTP polygons are supported")
    triangles = connectivity.reshape(-1, 3)
    alpha = np.fromstring(
        _data_array(root, "alpha.water").text or "", sep=" ", dtype=float
    )
    if alpha.size != points.shape[0]:
        raise ValueError(f"{path}: alpha.water is not point-based")
    return time, points, triangles, alpha, digest


def _connected_tip(
    points: np.ndarray,
    triangles: np.ndarray,
    alpha: np.ndarray,
    threshold: float,
    x_min: float,
    x_max: float,
    lid_z: float,
    search_top_z: float,
    seed_depth: float,
) -> tuple[float, int]:
    triangle_points = points[triangles]
    centers = triangle_points.mean(axis=1)
    cell_alpha = alpha[triangles].mean(axis=1)
    in_riser = (
        (centers[:, 0] >= x_min)
        & (centers[:, 0] <= x_max)
        & (centers[:, 2] >= lid_z - seed_depth)
        & (centers[:, 2] <= search_top_z)
    )
    wet_ids = np.flatnonzero(in_riser & (cell_alpha >= threshold))
    if wet_ids.size == 0:
        return lid_z, 0

    local_triangles = triangles[wet_ids]
    seeds = np.flatnonzero(
        triangle_points[wet_ids, :, 2].min(axis=1) <= lid_z + seed_depth
    )
    if seeds.size == 0:
        return lid_z, 0

    vertex_to_local: dict[int, list[int]] = defaultdict(list)
    for local_id, vertices in enumerate(local_triangles):
        for vertex in vertices:
            vertex_to_local[int(vertex)].append(local_id)

    visited = np.zeros(wet_ids.size, dtype=bool)
    queue: deque[int] = deque(int(seed) for seed in seeds)
    visited[seeds] = True
    while queue:
        local_id = queue.popleft()
        for vertex in local_triangles[local_id]:
            for neighbour in vertex_to_local[int(vertex)]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    queue.append(neighbour)

    connected_vertices = np.unique(local_triangles[visited].ravel())
    tip = float(np.max(points[connected_vertices, 2])) if connected_vertices.size else lid_z
    return max(tip, lid_z), int(np.count_nonzero(visited))


def _snapshot_paths(root: Path, filename: str) -> list[tuple[float, Path]]:
    paths = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            time = float(directory.name)
        except ValueError:
            continue
        path = directory / filename
        if path.is_file():
            paths.append((time, path))
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtp-root", type=Path, required=True)
    parser.add_argument("--filename", default="frontCentre.vtp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--time-shift-s", type=float, default=0.0)
    parser.add_argument("--min-solver-time-s", type=float, default=float("-inf"))
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--lid-z", type=float, required=True)
    parser.add_argument("--rim-z", type=float, required=True)
    parser.add_argument("--search-top-z", type=float, required=True)
    parser.add_argument("--seed-depth", type=float, default=0.02)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5])
    args = parser.parse_args()

    vtp_root = args.vtp_root.resolve()
    snapshots = [
        (time, path)
        for time, path in _snapshot_paths(vtp_root, args.filename)
        if time >= args.min_solver_time_s
    ]
    if not snapshots:
        raise RuntimeError(f"no matching VTP snapshots found under {vtp_root}")

    rows = []
    source_digest = hashlib.sha256()
    time_disagreement_max = 0.0
    for index, (directory_time, path) in enumerate(snapshots, start=1):
        field_time, points, triangles, alpha, file_hash = _read_vtp(path)
        time_disagreement_max = max(time_disagreement_max, abs(field_time - directory_time))
        source_digest.update(path.relative_to(vtp_root).as_posix().encode("utf-8"))
        source_digest.update(bytes.fromhex(file_hash))
        row: dict[str, float | int] = {
            "t_solver_s": field_time,
            "t_match_s": field_time + args.time_shift_s,
        }
        for threshold in args.thresholds:
            tip_z, connected_cells = _connected_tip(
                points,
                triangles,
                alpha,
                threshold,
                args.x_min,
                args.x_max,
                args.lid_z,
                args.search_top_z,
                args.seed_depth,
            )
            label = f"a{threshold:g}".replace(".", "p")
            row[f"column_height_{label}_m"] = max(
                min(tip_z, args.rim_z) - args.lid_z, 0.0
            )
            row[f"connected_tip_{label}_m"] = max(tip_z - args.lid_z, 0.0)
            row[f"connected_cells_{label}"] = connected_cells
        rows.append(row)
        if index % 25 == 0 or index == len(snapshots):
            print(f"processed {index}/{len(snapshots)} snapshots", flush=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": 1,
        "observer": (
            "alpha.water threshold on triangular cells; retain only the wet "
            "component sharing vertices with the riser-base seed band"
        ),
        "vtp_root": vtp_root.as_posix(),
        "filename": args.filename,
        "snapshot_count": len(rows),
        "clock": f"t_match=t_solver{args.time_shift_s:+g} s",
        "solver_time_coverage_s": [rows[0]["t_solver_s"], rows[-1]["t_solver_s"]],
        "matching_time_coverage_s": [rows[0]["t_match_s"], rows[-1]["t_match_s"]],
        "geometry_m": {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "lid_z": args.lid_z,
            "rim_z": args.rim_z,
            "search_top_z": args.search_top_z,
            "seed_depth": args.seed_depth,
        },
        "thresholds": args.thresholds,
        "maximum_directory_vs_field_time_difference_s": time_disagreement_max,
        "source_bundle_sha256": source_digest.hexdigest(),
        "output": output.as_posix(),
    }
    metadata_path = (
        args.metadata.resolve()
        if args.metadata is not None
        else output.with_suffix(".meta.json")
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
