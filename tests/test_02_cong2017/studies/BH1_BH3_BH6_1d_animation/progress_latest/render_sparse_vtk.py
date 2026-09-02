#!/usr/bin/env python3
"""Render immutable, already-written OpenFOAM checkpoints for the progress viewer."""
from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
VTK_ROOT = Path("/tmp/geysering-bh-progress-vtk-20260810")
OUT_ROOT = HERE / "frames_2d_extension"
MANIFEST = HERE / "frames_2d_extension.json"

PIPE_LENGTH = 6.59
PIPE_INVERT = -0.025
PIPE_CROWN = 0.025
RISER_X = 3.47
VALVE_X = 5.98
RIM_Z = 1.825
DOMAIN_TOP = 3.025
EXTERNAL_LEFT = 3.32
EXTERNAL_RIGHT = 3.62

CASES = {
    "BH1": {"Dr": 0.016},
    "BH3": {"Dr": 0.026},
    "BH6": {"Dr": 0.041},
}


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
    cells = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    if point_node is None or not {"connectivity", "offsets"}.issubset(cells):
        raise ValueError(f"Missing VTU geometry in {path}")
    points = array(point_node, 3)
    connectivity = array(cells["connectivity"]).astype(int)
    offsets = array(cells["offsets"]).astype(int)
    centres = np.empty((len(offsets), 2), dtype=float)
    begin = 0
    for index, end in enumerate(offsets):
        point_ids = connectivity[begin:end]
        centres[index] = points[point_ids][:, (0, 2)].mean(axis=0)
        begin = int(end)
    return time_s, alpha, centres


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


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


def colours(alpha: np.ndarray, valid: np.ndarray) -> np.ndarray:
    water = np.array([37.0, 99.0, 235.0])
    air = np.array([247.0, 249.0, 252.0])
    fraction = np.clip(alpha, 0.0, 1.0)[..., None]
    rgb = np.rint(air + fraction * (water - air)).astype(np.uint8)
    rgb[~valid] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def draw_h(image: np.ndarray, y: int, x0: int, x1: int, colour: tuple[int, int, int], thick: int = 2) -> None:
    height, width, _ = image.shape
    xa, xb = sorted((max(0, min(width - 1, x0)), max(0, min(width - 1, x1))))
    ya = max(0, y - thick // 2)
    yb = min(height, y + (thick + 1) // 2)
    image[ya:yb, xa : xb + 1] = colour


def draw_v(image: np.ndarray, x: int, y0: int, y1: int, colour: tuple[int, int, int], thick: int = 2) -> None:
    height, width, _ = image.shape
    ya, yb = sorted((max(0, min(height - 1, y0)), max(0, min(height - 1, y1))))
    xa = max(0, x - thick // 2)
    xb = min(width, x + (thick + 1) // 2)
    image[ya : yb + 1, xa:xb] = colour


def render_case(case_id: str, dr: float) -> list[dict[str, float | str]]:
    paths = sorted((VTK_ROOT / case_id).glob("*/internal.vtu"))
    if not paths:
        raise FileNotFoundError(f"No VTK checkpoints for {case_id}")

    parsed: list[tuple[float, Path]] = []
    for path in paths:
        time_s, _, _ = read_vtu(path, geometry=False)
        parsed.append((time_s, path))
    parsed.sort(key=lambda item: item[0])

    first_time, first_alpha, centres = read_vtu(parsed[0][1], geometry=True)
    assert centres is not None
    if len(first_alpha) != len(centres):
        raise ValueError(f"{case_id}: alpha field and mesh sizes differ")

    width, height = 1400, 650
    xmin, xmax = -0.08, PIPE_LENGTH + 0.08
    zmin, zmax = PIPE_INVERT - 0.055, DOMAIN_TOP + 0.055
    pxs = np.clip(((centres[:, 0] - xmin) / (xmax - xmin) * (width - 1)).astype(int), 0, width - 1)
    pys = np.clip(((zmax - centres[:, 1]) / (zmax - zmin) * (height - 1)).astype(int), 0, height - 1)
    flat = pys * width + pxs
    counts = np.bincount(flat, minlength=width * height).astype(float)
    valid = counts.reshape(height, width) > 0

    def px(x: float) -> int:
        return int(round((x - xmin) / (xmax - xmin) * (width - 1)))

    def py(z: float) -> int:
        return int(round((zmax - z) / (zmax - zmin) * (height - 1)))

    riser_width = dr * dr / 0.05
    left = RISER_X - 0.5 * riser_width
    right = RISER_X + 0.5 * riser_width
    output_dir = OUT_ROOT / case_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, float | str]] = []

    for frame_no, (hint_time, path) in enumerate(parsed):
        if frame_no == 0:
            time_s, alpha = first_time, first_alpha
        else:
            time_s, alpha, _ = read_vtu(path, geometry=False)
        if abs(time_s - hint_time) > 1.0e-6 or len(alpha) != len(centres):
            raise ValueError(f"{case_id}: inconsistent checkpoint {path}")

        sums = np.bincount(flat, weights=alpha, minlength=width * height)
        average = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).reshape(height, width)
        image = colours(average, valid)
        black, red, grey = (31, 41, 55), (220, 38, 38), (174, 184, 196)
        draw_h(image, py(PIPE_INVERT), px(0), px(PIPE_LENGTH), black)
        draw_h(image, py(PIPE_CROWN), px(0), px(left), black)
        draw_h(image, py(PIPE_CROWN), px(right), px(PIPE_LENGTH), black)
        draw_v(image, px(0), py(PIPE_INVERT), py(PIPE_CROWN), black)
        draw_v(image, px(PIPE_LENGTH), py(PIPE_INVERT), py(PIPE_CROWN), black)
        draw_v(image, px(left), py(PIPE_CROWN), py(RIM_Z), black)
        draw_v(image, px(right), py(PIPE_CROWN), py(RIM_Z), black)
        draw_h(image, py(RIM_Z), px(left) - 3, px(right) + 3, red)
        draw_v(image, px(EXTERNAL_LEFT), py(RIM_Z), py(DOMAIN_TOP), grey, 1)
        draw_v(image, px(EXTERNAL_RIGHT), py(RIM_Z), py(DOMAIN_TOP), grey, 1)
        draw_h(image, py(DOMAIN_TOP), px(EXTERNAL_LEFT), px(EXTERNAL_RIGHT), grey, 1)
        valve_x = px(VALVE_X)
        for yy in range(min(py(PIPE_CROWN), py(PIPE_INVERT)), max(py(PIPE_CROWN), py(PIPE_INVERT)) + 1, 5):
            draw_v(image, valve_x, yy, min(yy + 2, height - 1), black, 1)

        time_tag = f"{time_s:.2f}".replace(".", "p")
        output = output_dir / f"full_{time_tag}.png"
        write_png(output, image)
        manifest.append({
            "time": round(time_s, 8),
            "file": output.relative_to(HERE).as_posix(),
            "source": "immutable written OpenFOAM checkpoint",
        })
        print(f"{case_id}: rendered {time_s:.2f} s")
    return manifest


def main() -> None:
    payload = {
        case_id: render_case(case_id, float(spec["Dr"]))
        for case_id, spec in CASES.items()
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(MANIFEST)


if __name__ == "__main__":
    main()
