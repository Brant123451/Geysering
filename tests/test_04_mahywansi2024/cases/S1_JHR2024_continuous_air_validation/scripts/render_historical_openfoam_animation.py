#!/usr/bin/env python3
"""Render historical OpenFOAM alpha.water fields without promoting them.

The input is an existing ASCII OpenFOAM case.  Geometry is reconstructed from
``constant/polyMesh`` and every frame is read from the saved ``alpha.water``
and ``U`` fields.  The renderer is deliberately evidence-only: it never runs
OpenFOAM, changes a case, or creates an acceptance marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_initial_water_air import (
    AIR,
    MUTED,
    OUTLINE,
    TEXT,
    WATER,
    WHITE,
    _font,
    draw_panel,
)


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\r\n]*", "", text)


def _list_body(path: Path) -> tuple[int, str]:
    text = _without_comments(path.read_text(encoding="utf-8", errors="strict"))
    match = re.search(r"(?:^|\n)\s*(\d+)\s*\n\s*\((.*)\)\s*;?\s*$", text, re.DOTALL)
    if match is None:
        raise ValueError(f"OpenFOAM list was not found in {path}")
    return int(match.group(1)), match.group(2)


def read_points(path: Path) -> np.ndarray:
    count, body = _list_body(path)
    rows = re.findall(rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)", body)
    points = np.asarray(rows, dtype=float)
    if points.shape != (count, 3):
        raise ValueError(f"point count mismatch in {path}: {points.shape} != {(count, 3)}")
    return points


def read_faces(path: Path) -> list[np.ndarray]:
    count, body = _list_body(path)
    faces: list[np.ndarray] = []
    for width, labels in re.findall(r"(\d+)\s*\(([^()]*)\)", body):
        row = np.fromstring(labels, dtype=np.int64, sep=" ")
        if len(row) != int(width):
            raise ValueError(f"face width mismatch in {path}")
        faces.append(row)
    if len(faces) != count:
        raise ValueError(f"face count mismatch in {path}: {len(faces)} != {count}")
    return faces


def read_labels(path: Path) -> np.ndarray:
    count, body = _list_body(path)
    values = np.fromstring(body, dtype=np.int64, sep=" ")
    if len(values) != count:
        raise ValueError(f"label count mismatch in {path}: {len(values)} != {count}")
    return values


def read_scalar_field(path: Path) -> np.ndarray:
    text = _without_comments(path.read_text(encoding="utf-8", errors="strict"))
    uniform = re.search(rf"internalField\s+uniform\s+({FLOAT})\s*;", text)
    if uniform is not None:
        raise ValueError(f"uniform fields need an explicit cell count: {path}")
    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"nonuniform scalar internalField was not found in {path}")
    values = np.fromstring(match.group(2), dtype=float, sep=" ")
    count = int(match.group(1))
    if len(values) != count:
        raise ValueError(f"scalar count mismatch in {path}: {len(values)} != {count}")
    return values


def read_vector_field(path: Path) -> np.ndarray:
    text = _without_comments(path.read_text(encoding="utf-8", errors="strict"))
    match = re.search(
        r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if match is None:
        uniform = re.search(
            rf"internalField\s+uniform\s+\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)\s*;",
            text,
        )
        if uniform is None:
            raise ValueError(f"vector internalField was not found in {path}")
        return np.asarray([[float(uniform.group(i)) for i in range(1, 4)]])
    rows = re.findall(rf"\(\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*\)", match.group(2))
    values = np.asarray(rows, dtype=float)
    count = int(match.group(1))
    if values.shape != (count, 3):
        raise ValueError(f"vector count mismatch in {path}: {values.shape} != {(count, 3)}")
    return values


def reconstruct_cell_bounds(case: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = case / "constant" / "polyMesh"
    points = read_points(mesh / "points")
    faces = read_faces(mesh / "faces")
    owner = read_labels(mesh / "owner")
    neighbour = read_labels(mesh / "neighbour")
    if len(owner) != len(faces):
        raise ValueError("owner and face counts differ")
    cell_count = int(max(owner.max(), neighbour.max())) + 1
    xmin = np.full(cell_count, math.inf)
    xmax = np.full(cell_count, -math.inf)
    zmin = np.full(cell_count, math.inf)
    zmax = np.full(cell_count, -math.inf)

    def include(cell: int, face: np.ndarray) -> None:
        xyz = points[face]
        xmin[cell] = min(xmin[cell], float(xyz[:, 0].min()))
        xmax[cell] = max(xmax[cell], float(xyz[:, 0].max()))
        zmin[cell] = min(zmin[cell], float(xyz[:, 2].min()))
        zmax[cell] = max(zmax[cell], float(xyz[:, 2].max()))

    for index, face in enumerate(faces):
        include(int(owner[index]), face)
        if index < len(neighbour):
            include(int(neighbour[index]), face)
    bounds = np.column_stack((xmin, xmax, zmin, zmax))
    if not np.isfinite(bounds).all():
        raise ValueError("non-finite reconstructed cell bounds")
    return points, bounds


def numeric_times(case: Path, end_time: float | None) -> list[tuple[float, Path]]:
    result: list[tuple[float, Path]] = []
    for child in case.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if end_time is not None and value > end_time + 1e-12:
            continue
        if (child / "alpha.water").is_file() and (child / "U").is_file():
            result.append((value, child))
    return sorted(result)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_frame(
    bounds: np.ndarray,
    alpha: np.ndarray,
    time_s: float,
    umax: float,
    output: Path,
) -> None:
    canvas = Image.new("RGB", (1600, 1600), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((105, 34), "Mahyawansi JHR2024 — historical 2D field", font=_font(38, True), fill=TEXT)
    draw.text(
        (107, 88),
        f"coarse grid | Stage 1 (no air injection) | t = {time_s:.2f} s | max |U| = {umax:.4g} m/s",
        font=_font(23),
        fill=MUTED,
    )
    draw.rectangle((102, 133, 1498, 180), fill=(255, 238, 224), outline=(178, 70, 28), width=2)
    draw.text(
        (120, 143),
        "HISTORICAL FAILED DIAGNOSTIC — NOT AN ACCEPTED RESULT",
        font=_font(24, True),
        fill=(145, 52, 20),
    )

    draw.rectangle((1110, 54, 1148, 84), fill=WATER, outline=OUTLINE, width=2)
    draw.text((1160, 54), "water", font=_font(21), fill=TEXT)
    draw.rectangle((1282, 54, 1320, 84), fill=AIR, outline=OUTLINE, width=2)
    draw.text((1332, 54), "air", font=_font(21), fill=TEXT)

    draw_panel(
        canvas,
        bounds,
        alpha,
        box=(105, 240, 1495, 975),
        domain=(-1.84, 1.31, -0.04, 2.04),
        title="A   Full pipe and external-air domain",
        annotations="full",
    )
    draw_panel(
        canvas,
        bounds,
        alpha,
        box=(105, 1100, 1495, 1495),
        domain=(-1.62, 0.18, -0.035, 0.70),
        title="B   Air-supply branch, main pipe and riser (enlarged)",
        annotations="zoom",
    )
    raw_min = float(alpha.min())
    raw_max = float(alpha.max())
    draw.text(
        (105, 1560),
        f"Raw alpha.water range = [{raw_min:.8g}, {raw_max:.8g}]. Display classification: water >= 0.5; air < 0.5.",
        font=_font(18),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end-time", type=float)
    args = parser.parse_args()

    case = args.case.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    points, bounds = reconstruct_cell_bounds(case)
    times = numeric_times(case, args.end_time)
    if not times:
        raise SystemExit("no saved alpha.water/U time directories were found")

    frame_records: list[dict[str, object]] = []
    expected_cells = len(bounds)
    for index, (time_s, time_dir) in enumerate(times):
        alpha_path = time_dir / "alpha.water"
        velocity_path = time_dir / "U"
        alpha = read_scalar_field(alpha_path)
        velocity = read_vector_field(velocity_path)
        if len(alpha) != expected_cells:
            raise ValueError(f"{time_dir}: alpha has {len(alpha)} cells, expected {expected_cells}")
        umax = float(np.linalg.norm(velocity, axis=1).max())
        frame_path = output / "frames" / f"frame_{index:04d}.png"
        render_frame(bounds, alpha, time_s, umax, frame_path)
        frame_records.append(
            {
                "frame": index,
                "time_s": time_s,
                "time_directory": str(time_dir),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "umax_m_per_s": umax,
                "alpha_sha256": sha256(alpha_path),
                "U_sha256": sha256(velocity_path),
                "png": str(frame_path),
                "png_sha256": sha256(frame_path),
            }
        )
        print(f"rendered {index + 1}/{len(times)}: t={time_s:g} s")

    manifest = {
        "schema_version": 1,
        "evidence_class": "historical_failed_diagnostic_visualization_only",
        "accepted_result": False,
        "physical_condition": "Mahyawansi_JHR2024_continuous_air_S1",
        "grid_level": "coarse",
        "stage": "Stage 1 simple water flow; no 5700 Pa air injection",
        "known_limitation": "run later failed the open-boundary acoustic stability gate",
        "source_case": str(case),
        "cell_count": expected_cells,
        "point_count": len(points),
        "time_start_s": times[0][0],
        "time_end_s": times[-1][0],
        "frame_count": len(frame_records),
        "rendering": {
            "phase_threshold": "water if alpha.water >= 0.5; air otherwise",
            "raw_fields_are_not_clipped_or_modified": True,
        },
        "frames": frame_records,
    }
    manifest_path = output / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
