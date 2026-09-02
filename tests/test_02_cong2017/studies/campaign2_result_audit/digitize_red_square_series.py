#!/usr/bin/env python3
"""Digitize red square markers from a fixed-axis paper figure raster.

This is deliberately a narrow evidence helper, not a general plot reader.  It
records the raster hash, axis calibration, colour rule, and marker-column rule
so values derived from Cong et al. (2017) Fig. 7(d) remain auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--bottom", type=int, required=True)
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--minimum-red-pixels-per-column", type=int, default=7)
    parser.add_argument("--minimum-marker-width-px", type=int, default=3)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    red = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    blue = image[:, :, 2].astype(float)
    mask = (
        (red > 150.0)
        & (red > 1.5 * green)
        & (red > 1.5 * blue)
        & (green < 150.0)
    )
    yy, xx = np.indices(mask.shape)
    mask &= (
        (xx >= args.left)
        & (xx <= args.right)
        & (yy >= args.top)
        & (yy <= args.bottom)
    )
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("no red pixels in calibrated axes")

    counts = np.bincount(xs, minlength=image.shape[1])
    dense_columns = np.flatnonzero(counts >= args.minimum_red_pixels_per_column)
    groups: list[list[int]] = []
    for column in dense_columns:
        if not groups or column > groups[-1][-1] + 1:
            groups.append([int(column)])
        else:
            groups[-1].append(int(column))

    points: list[dict[str, float]] = []
    for group in groups:
        if len(group) < args.minimum_marker_width_px:
            continue
        selected = (xs >= group[0]) & (xs <= group[-1])
        x_pixel = float(np.median(xs[selected]))
        y_pixel = float(np.median(ys[selected]))
        x_value = args.x_min + (x_pixel - args.left) * (
            args.x_max - args.x_min
        ) / (args.right - args.left)
        y_value = args.y_max - (y_pixel - args.top) * (
            args.y_max - args.y_min
        ) / (args.bottom - args.top)
        points.append(
            {
                "x": x_value,
                "y": y_value,
                "x_pixel": x_pixel,
                "y_pixel": y_pixel,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "x_pixel", "y_pixel"])
        writer.writeheader()
        writer.writerows(points)

    values = np.asarray([point["y"] for point in points], dtype=float)
    payload = {
        "schema_version": 1,
        "source_image": str(args.image),
        "source_image_sha256": file_sha256(args.image),
        "render_contract": "Cong et al. (2017) offprint, PDF page 7, 150 dpi PNG",
        "axes_pixels": {
            "left": args.left,
            "right": args.right,
            "top": args.top,
            "bottom": args.bottom,
        },
        "axes_values": {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
        },
        "colour_rule": "R>150, R>1.5G, R>1.5B, G<150",
        "marker_rule": {
            "minimum_red_pixels_per_column": args.minimum_red_pixels_per_column,
            "minimum_marker_width_px": args.minimum_marker_width_px,
        },
        "point_count": len(points),
        "y_statistics": {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        },
        "output_csv": str(args.output_csv),
        "qualification": "figure-digitized approximation, not a tabulated measurement",
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
