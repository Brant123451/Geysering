#!/usr/bin/env python3
"""Render the actual OpenFOAM t=0 alpha.water field exported by foamToVTK.

The script intentionally reads cell data and mesh coordinates from the VTU
file.  It does not reconstruct the phase layout from the case dimensions.
Only NumPy and Pillow are required.
"""

from __future__ import annotations

import argparse
import base64
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DTYPES = {
    "Float32": np.dtype("<f4"),
    "Float64": np.dtype("<f8"),
    "Int32": np.dtype("<i4"),
    "Int64": np.dtype("<i8"),
    "UInt8": np.dtype("u1"),
    "UInt32": np.dtype("<u4"),
    "UInt64": np.dtype("<u8"),
}

WATER = (43, 131, 207)
AIR = (226, 232, 238)
OUTLINE = (43, 53, 63)
GRID = (218, 224, 230)
TEXT = (31, 39, 48)
MUTED = (92, 103, 115)
WHITE = (255, 255, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _decode(array: ET.Element, header_type: str) -> np.ndarray:
    if array.attrib.get("format") != "binary":
        raise ValueError(f"Expected inline binary VTU data for {array.attrib.get('Name')}")
    encoded = "".join((array.text or "").split())
    payload = base64.b64decode(encoded)
    header_dtype = DTYPES[header_type]
    header_size = header_dtype.itemsize
    if len(payload) < header_size:
        raise ValueError(f"Truncated VTU array: {array.attrib.get('Name')}")
    byte_count = int(np.frombuffer(payload[:header_size], dtype=header_dtype, count=1)[0])
    raw = payload[header_size : header_size + byte_count]
    if len(raw) != byte_count:
        raise ValueError(f"VTU byte-count mismatch for {array.attrib.get('Name')}")
    return np.frombuffer(raw, dtype=DTYPES[array.attrib["type"]]).copy()


def read_vtu(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError("VTU Piece was not found")
    header_type = root.attrib.get("header_type", "UInt32")

    points_node = piece.find("./Points/DataArray")
    connectivity_node = piece.find("./Cells/DataArray[@Name='connectivity']")
    offsets_node = piece.find("./Cells/DataArray[@Name='offsets']")
    alpha_node = piece.find("./CellData/DataArray[@Name='alpha.water']")
    if any(node is None for node in (points_node, connectivity_node, offsets_node, alpha_node)):
        raise ValueError("Required Points/Cells/alpha.water arrays were not found")

    points = _decode(points_node, header_type).reshape(-1, 3)
    connectivity = _decode(connectivity_node, header_type).astype(np.int64, copy=False)
    offsets = _decode(offsets_node, header_type).astype(np.int64, copy=False)
    alpha = _decode(alpha_node, header_type)

    starts = np.concatenate(([0], offsets[:-1]))
    widths = offsets - starts
    if len(alpha) != len(offsets):
        raise ValueError(f"Cell count mismatch: alpha={len(alpha)}, offsets={len(offsets)}")

    if np.all(widths == widths[0]):
        cell_points = points[connectivity.reshape(len(offsets), int(widths[0]))]
        xmin = cell_points[:, :, 0].min(axis=1)
        xmax = cell_points[:, :, 0].max(axis=1)
        zmin = cell_points[:, :, 2].min(axis=1)
        zmax = cell_points[:, :, 2].max(axis=1)
    else:
        xmin = np.empty(len(offsets))
        xmax = np.empty(len(offsets))
        zmin = np.empty(len(offsets))
        zmax = np.empty(len(offsets))
        for i, (start, stop) in enumerate(zip(starts, offsets)):
            xyz = points[connectivity[start:stop]]
            xmin[i], xmax[i] = xyz[:, 0].min(), xyz[:, 0].max()
            zmin[i], zmax[i] = xyz[:, 2].min(), xyz[:, 2].max()

    bounds = np.column_stack((xmin, xmax, zmin, zmax))
    return points, bounds, alpha


def _nice_ticks(vmin: float, vmax: float, target: int = 7) -> list[float]:
    span = max(vmax - vmin, 1e-12)
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    scaled = raw / magnitude
    step = (1 if scaled <= 1 else 2 if scaled <= 2 else 2.5 if scaled <= 2.5 else 5 if scaled <= 5 else 10) * magnitude
    first = math.ceil(vmin / step - 1e-10) * step
    ticks = []
    value = first
    while value <= vmax + 1e-10:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _tick_label(value: float, span: float) -> str:
    decimals = 2 if span <= 1.0 else 1
    return f"{value:.{decimals}f}"


def draw_panel(
    image: Image.Image,
    cells: np.ndarray,
    alpha: np.ndarray,
    box: tuple[int, int, int, int],
    domain: tuple[float, float, float, float],
    title: str,
    annotations: str | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    xmin, xmax, zmin, zmax = domain
    width, height = right - left, bottom - top
    scale = min(width / (xmax - xmin), height / (zmax - zmin))
    used_w = (xmax - xmin) * scale
    used_h = (zmax - zmin) * scale
    x0 = left + (width - used_w) / 2
    y0 = top + (height - used_h) / 2

    def px(x: float) -> int:
        return int(round(x0 + (x - xmin) * scale))

    def py(z: float) -> int:
        return int(round(y0 + (zmax - z) * scale))

    plot_l, plot_r = px(xmin), px(xmax)
    plot_t, plot_b = py(zmax), py(zmin)

    x_ticks = _nice_ticks(xmin, xmax)
    z_ticks = _nice_ticks(zmin, zmax)
    tick_font = _font(21)
    label_font = _font(24)
    panel_font = _font(28, bold=True)
    note_font = _font(21)

    for value in x_ticks:
        xx = px(value)
        draw.line((xx, plot_t, xx, plot_b), fill=GRID, width=1)
    for value in z_ticks:
        yy = py(value)
        draw.line((plot_l, yy, plot_r, yy), fill=GRID, width=1)

    phase = Image.new("RGB", image.size, WHITE)
    phase_draw = ImageDraw.Draw(phase)
    occupancy = Image.new("L", image.size, 0)
    occupancy_draw = ImageDraw.Draw(occupancy)

    visible = (
        (cells[:, 1] >= xmin)
        & (cells[:, 0] <= xmax)
        & (cells[:, 3] >= zmin)
        & (cells[:, 2] <= zmax)
    )
    for (cx0, cx1, cz0, cz1), aw in zip(cells[visible], alpha[visible]):
        rect = (px(float(cx0)) - 1, py(float(cz1)) - 1, px(float(cx1)) + 1, py(float(cz0)) + 1)
        fill = WATER if aw >= 0.5 else AIR
        phase_draw.rectangle(rect, fill=fill)
        occupancy_draw.rectangle(rect, fill=255)

    image.paste(phase, mask=occupancy)
    dilated = occupancy.filter(ImageFilter.MaxFilter(5))
    eroded = occupancy.filter(ImageFilter.MinFilter(5))
    edge = np.asarray(dilated, dtype=np.int16) - np.asarray(eroded, dtype=np.int16)
    edge_mask = Image.fromarray(np.where(edge > 0, 255, 0).astype(np.uint8), mode="L")
    outline_layer = Image.new("RGB", image.size, OUTLINE)
    image.paste(outline_layer, mask=edge_mask)
    draw = ImageDraw.Draw(image)

    draw.line((plot_l, plot_b, plot_r, plot_b), fill=OUTLINE, width=2)
    draw.line((plot_l, plot_t, plot_l, plot_b), fill=OUTLINE, width=2)
    span_x, span_z = xmax - xmin, zmax - zmin
    for value in x_ticks:
        xx = px(value)
        draw.line((xx, plot_b, xx, plot_b + 8), fill=OUTLINE, width=2)
        label = _tick_label(value, span_x)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((xx - (bbox[2] - bbox[0]) / 2, plot_b + 11), label, font=tick_font, fill=TEXT)
    for value in z_ticks:
        yy = py(value)
        draw.line((plot_l - 8, yy, plot_l, yy), fill=OUTLINE, width=2)
        label = _tick_label(value, span_z)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_l - 14 - (bbox[2] - bbox[0]), yy - (bbox[3] - bbox[1]) / 2), label, font=tick_font, fill=TEXT)

    title_bbox = draw.textbbox((0, 0), title, font=panel_font)
    draw.text((plot_l, plot_t - 48), title, font=panel_font, fill=TEXT)
    draw.text(((plot_l + plot_r) / 2 - 28, plot_b + 48), "x (m)", font=label_font, fill=TEXT)
    z_label = Image.new("RGBA", (160, 45), (255, 255, 255, 0))
    z_draw = ImageDraw.Draw(z_label)
    z_draw.text((0, 0), "z (m)", font=label_font, fill=TEXT)
    z_label = z_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(z_label, (plot_l - 105, int((plot_t + plot_b - z_label.height) / 2)), z_label)

    if annotations:
        water_level = 0.5842
        yy = py(water_level)
        draw.line((px(-0.07), yy, px(0.075), yy), fill=(17, 90, 155), width=4)
        if annotations == "zoom":
            draw.line((px(-0.34), py(0.625), px(-0.06), yy), fill=(17, 90, 155), width=2)
            draw.text((px(-0.66), py(0.67)), "initial water level  z = 0.5842 m", font=note_font, fill=TEXT)
        else:
            draw.text((px(0.085), yy - 16), "initial water level  z = 0.5842 m", font=note_font, fill=TEXT)

        inlet_x = -1.50
        inlet_top = 0.15
        ax, ay = px(inlet_x), py(inlet_top)
        draw.line((ax, ay - 50, ax, ay - 8), fill=OUTLINE, width=3)
        draw.polygon(((ax, ay), (ax - 7, ay - 12), (ax + 7, ay - 12)), fill=OUTLINE)
        draw.text((ax - 125, ay - 82), "air inlet", font=note_font, fill=TEXT)

        riser_x = px(0.0)
        riser_z = 0.43 if annotations == "zoom" else 0.98
        draw.line((riser_x + 18, py(riser_z), riser_x + 62, py(riser_z)), fill=OUTLINE, width=2)
        draw.text((riser_x + 70, py(riser_z) - 17), "riser", font=note_font, fill=TEXT)


def render(vtu: Path, output: Path) -> dict[str, float | int]:
    points, cells, alpha = read_vtu(vtu)
    cell_count = len(alpha)
    water_cells = int(np.count_nonzero(alpha >= 0.5))
    air_cells = cell_count - water_cells

    canvas = Image.new("RGB", (2100, 2240), WHITE)
    draw = ImageDraw.Draw(canvas)
    title_font = _font(40, bold=True)
    subtitle_font = _font(23)
    legend_font = _font(23)

    draw.text((135, 50), "Actual OpenFOAM initial water-air field  (t = 0 s)", font=title_font, fill=TEXT)
    subtitle = f"alpha.water from VTU cell data  |  {cell_count:,} cells  |  quasi-2D mesh"
    draw.text((137, 106), subtitle, font=subtitle_font, fill=MUTED)

    draw.rectangle((1390, 57, 1432, 91), fill=WATER, outline=OUTLINE, width=2)
    draw.text((1448, 58), "Water  (alpha.water = 1)", font=legend_font, fill=TEXT)
    draw.rectangle((1390, 103, 1432, 137), fill=AIR, outline=OUTLINE, width=2)
    draw.text((1448, 104), "Air  (alpha.water = 0)", font=legend_font, fill=TEXT)

    draw_panel(
        canvas,
        cells,
        alpha,
        box=(150, 220, 1950, 1245),
        domain=(-1.84, 1.31, -0.04, 2.04),
        title="A   Full computational domain",
        annotations="full",
    )
    draw_panel(
        canvas,
        cells,
        alpha,
        box=(150, 1415, 1950, 2075),
        domain=(-1.62, 0.18, -0.035, 0.70),
        title="B   Main pipe, air-inlet branch and T-junction (enlarged)",
        annotations="zoom",
    )

    footer = (
        f"VTU bounds: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] m, "
        f"z=[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] m  |  "
        f"water cells={water_cells:,}, air cells={air_cells:,}"
    )
    draw.text((150, 2188), footer, font=_font(20), fill=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True, dpi=(180, 180))
    return {
        "cells": cell_count,
        "water_cells": water_cells,
        "air_cells": air_cells,
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vtu", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stats = render(args.vtu, args.output)
    print(args.output.resolve())
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
