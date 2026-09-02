#!/usr/bin/env python3
"""Rasterize the completed B-H3 OpenFOAM alpha.water VTU series.

The frame colours come directly from cell-centred alpha.water.  No temporal
alignment, interface repainting, or outcome fitting is applied.
"""
from __future__ import annotations

import csv
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
VTK_ROOT = HERE / "openfoam_2d" / "VTK_BH3_HTML"
OUT = HERE / "openfoam_2d" / "frames"
MANIFEST = HERE / "openfoam_2d" / "frames.json"
LEVELS = HERE.parent / "results" / "openfoam_2d_riser_series.csv"
PT1 = HERE.parent / "results" / "openfoam_2d_pt1_series.csv"

PIPE_LENGTH = 6.59
PIPE_INVERT = -0.025
PIPE_CROWN = 0.025
RISER_X = 3.47
RISER_WIDTH = 0.01352
RIM_Z = 1.825
DOMAIN_TOP = 3.025
EXTERNAL_LEFT = 3.32
EXTERNAL_RIGHT = 3.62


def array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Expected ASCII foamToVTK output")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def read_vtu(path: Path, geometry: bool) -> tuple[float, np.ndarray, np.ndarray | None]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    if time_node is None or alpha_node is None:
        raise ValueError(f"Missing TimeValue or alpha.water in {path}")
    time_s = float((time_node.text or "0").strip())
    alpha = np.clip(array(alpha_node), 0.0, 1.0)
    if not geometry:
        return time_s, alpha, None

    point_node = root.find(".//Points/DataArray")
    cell_nodes = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    if point_node is None or not {"connectivity", "offsets"}.issubset(cell_nodes):
        raise ValueError(f"Missing VTU geometry in {path}")
    points = array(point_node, 3)
    connectivity = array(cell_nodes["connectivity"]).astype(int)
    offsets = array(cell_nodes["offsets"]).astype(int)
    centres = np.empty((len(offsets), 2), dtype=float)
    begin = 0
    for index, end in enumerate(offsets):
        cell = connectivity[begin:end]
        centres[index] = points[cell][:, (0, 2)].mean(axis=0)
        begin = int(end)
    return time_s, alpha, centres


def vtk_paths() -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for path in VTK_ROOT.glob("*/internal.vtu"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.read(1024)
        match = re.search(r"\btime='([^']+)'", header)
        if match:
            found.append((float(match.group(1)), path))
        else:
            root = ET.parse(path).getroot()
            node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
            if node is None:
                raise ValueError(f"Cannot find VTU time: {path}")
            found.append((float((node.text or "0").strip()), path))
    found.sort(key=lambda item: item[0])
    if len(found) < 260 or found[0][0] > 1.0e-8 or found[-1][0] < 12.999:
        raise RuntimeError(
            f"Expected a complete 0--13 s VTU series (about 261 frames), found {len(found)} "
            f"from {found[0][0] if found else None} to {found[-1][0] if found else None}"
        )
    return found


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8:
        raise ValueError("PNG input must be uint8 RGB")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def colours(alpha: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    water = np.array([38.0, 119.0, 218.0])
    air = np.array([247.0, 249.0, 252.0])
    a = np.clip(alpha, 0.0, 1.0)[..., None]
    rgb = np.rint(air + a * (water - air)).astype(np.uint8)
    if valid is not None:
        rgb[~valid] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def draw_h(rgb: np.ndarray, y: int, x0: int, x1: int, colour: tuple[int, int, int], thick: int = 2) -> None:
    h, w, _ = rgb.shape
    xa, xb = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    ya, yb = max(0, y - thick // 2), min(h, y + (thick + 1) // 2)
    rgb[ya:yb, xa:xb + 1] = colour


def draw_v(rgb: np.ndarray, x: int, y0: int, y1: int, colour: tuple[int, int, int], thick: int = 2) -> None:
    h, w, _ = rgb.shape
    ya, yb = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    xa, xb = max(0, x - thick // 2), min(w, x + (thick + 1) // 2)
    rgb[ya:yb + 1, xa:xb] = colour


def read_series(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.asarray([float(row[name]) for row in rows], dtype=float) for name in rows[0]}


def finite_interp(series: dict[str, np.ndarray], time_key: str, value_key: str, t: float, default: float = 0.0) -> float:
    x, y = series[time_key], series[value_key]
    good = np.isfinite(x) & np.isfinite(y)
    return default if not np.any(good) else float(np.interp(t, x[good], y[good]))


def main() -> None:
    paths = vtk_paths()
    first_time, first_alpha, centres = read_vtu(paths[0][1], geometry=True)
    assert centres is not None
    if len(first_alpha) != len(centres):
        raise ValueError("alpha.water length does not match VTU cells")
    levels = read_series(LEVELS)
    pt1 = read_series(PT1) if PT1.exists() else None
    OUT.mkdir(parents=True, exist_ok=True)

    # Full apparatus raster mapping, including the above-rim external atmosphere.
    fw, fh = 1400, 650
    xmin, xmax = -0.08, PIPE_LENGTH + 0.08
    zmin, zmax = PIPE_INVERT - 0.055, DOMAIN_TOP + 0.055
    fx = np.clip(((centres[:, 0] - xmin) / (xmax - xmin) * (fw - 1)).astype(int), 0, fw - 1)
    fy = np.clip(((zmax - centres[:, 1]) / (zmax - zmin) * (fh - 1)).astype(int), 0, fh - 1)
    flat = fy * fw + fx
    full_counts = np.bincount(flat, minlength=fw * fh).astype(float)

    # Riser-only structured mapping for an enlarged but unaltered alpha field.
    left = RISER_X - 0.5 * RISER_WIDTH
    right = RISER_X + 0.5 * RISER_WIDTH
    riser = (
        (centres[:, 0] >= left - 1e-9) & (centres[:, 0] <= right + 1e-9)
        & (centres[:, 1] >= PIPE_CROWN - 1e-9) & (centres[:, 1] <= RIM_Z + 1e-9)
    )
    xvals, xindex = np.unique(np.round(centres[riser, 0], 9), return_inverse=True)
    zvals, zindex = np.unique(np.round(centres[riser, 1], 9), return_inverse=True)
    riser_flat = zindex * len(xvals) + xindex
    zw, zh = 300, 700
    xpick = np.rint(np.linspace(0, len(xvals) - 1, zw)).astype(int)
    zpick = np.rint(np.linspace(len(zvals) - 1, 0, zh)).astype(int)

    def px(x: float) -> int:
        return int(round((x - xmin) / (xmax - xmin) * (fw - 1)))

    def py(z: float) -> int:
        return int(round((zmax - z) / (zmax - zmin) * (fh - 1)))

    manifest: list[dict[str, float | str]] = []
    for frame_no, (hint_time, path) in enumerate(paths):
        if frame_no == 0:
            time_s, alpha = first_time, first_alpha
        else:
            time_s, alpha, _ = read_vtu(path, geometry=False)
        if abs(time_s - hint_time) > 1.0e-6:
            raise ValueError(f"Time mismatch in {path}")

        sums = np.bincount(flat, weights=alpha, minlength=fw * fh)
        avg = np.divide(sums, full_counts, out=np.zeros_like(sums), where=full_counts > 0).reshape(fh, fw)
        full = colours(avg, full_counts.reshape(fh, fw) > 0)
        black, red, grey = (38, 52, 66), (210, 54, 48), (177, 186, 196)
        draw_h(full, py(PIPE_INVERT), px(0), px(PIPE_LENGTH), black)
        draw_h(full, py(PIPE_CROWN), px(0), px(left), black)
        draw_h(full, py(PIPE_CROWN), px(right), px(PIPE_LENGTH), black)
        draw_v(full, px(0), py(PIPE_INVERT), py(PIPE_CROWN), black)
        draw_v(full, px(PIPE_LENGTH), py(PIPE_INVERT), py(PIPE_CROWN), black)
        draw_v(full, px(left), py(PIPE_CROWN), py(RIM_Z), black)
        draw_v(full, px(right), py(PIPE_CROWN), py(RIM_Z), black)
        draw_h(full, py(RIM_Z), px(left) - 3, px(right) + 3, red)
        draw_v(full, px(EXTERNAL_LEFT), py(RIM_Z), py(DOMAIN_TOP), grey, 1)
        draw_v(full, px(EXTERNAL_RIGHT), py(RIM_Z), py(DOMAIN_TOP), grey, 1)
        draw_h(full, py(DOMAIN_TOP), px(EXTERNAL_LEFT), px(EXTERNAL_RIGHT), grey, 1)
        # Dashed valve marker.
        valve_x = px(5.98)
        for yy in range(min(py(PIPE_CROWN), py(PIPE_INVERT)), max(py(PIPE_CROWN), py(PIPE_INVERT)) + 1, 5):
            draw_v(full, valve_x, yy, min(yy + 2, fh - 1), black, 1)

        rgrid = np.zeros((len(zvals), len(xvals)), dtype=float)
        rcounts = np.bincount(riser_flat, minlength=rgrid.size).reshape(rgrid.shape)
        rsum = np.bincount(riser_flat, weights=alpha[riser], minlength=rgrid.size).reshape(rgrid.shape)
        np.divide(rsum, rcounts, out=rgrid, where=rcounts > 0)
        zoom = colours(rgrid[np.ix_(zpick, xpick)])
        draw_v(zoom, 0, 0, zh - 1, black)
        draw_v(zoom, zw - 1, 0, zh - 1, black)
        draw_h(zoom, 0, 0, zw - 1, red)
        draw_h(zoom, zh - 1, 0, zw - 1, black)

        full_path = OUT / f"full_{frame_no:04d}.png"
        zoom_path = OUT / f"zoom_{frame_no:04d}.png"
        write_png(full_path, full)
        write_png(zoom_path, zoom)
        yfs = finite_interp(levels, "t_s", "Yfs_m_above_crown", time_s)
        yint = finite_interp(levels, "t_s", "Yint_m_above_crown", time_s)
        head = 0.0 if pt1 is None else finite_interp(pt1, "t_s", "head_m_water", time_s)
        manifest.append({
            "file": f"openfoam_2d/frames/{full_path.name}",
            "riserFile": f"openfoam_2d/frames/{zoom_path.name}",
            "time": time_s,
            "Yfs": yfs,
            "Yint": yint,
            "head": head,
        })
        if frame_no % 20 == 0 or frame_no == len(paths) - 1:
            print(f"Rendered {frame_no + 1}/{len(paths)} at t={time_s:.2f} s", flush=True)

    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
