#!/usr/bin/env python3
"""Measure the resolved wet plume above the local Case B tower mouth.

Input must be ASCII ``internal.vtu`` files produced after reconstruction by
``foamToVTK -ascii -no-point-data``.  The local calculation starts at source
time 6.5 s, so every result records both time coordinates explicitly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VTK = CASE_DIR / "VTK_TOP_PLUME_DIAGNOSTIC"
DEFAULT_OUTPUT = CASE_DIR / "outputs" / "plume_diagnostics"


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_VTK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-time-offset", type=float, default=6.5)
    parser.add_argument("--thresholds", type=comma_floats, default=[0.01, 0.5])
    parser.add_argument(
        "--levels-above-rim",
        type=comma_floats,
        default=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
    )
    parser.add_argument("--axis-x", type=float, default=0.0)
    parser.add_argument("--box-xmin", type=float, default=-0.12)
    parser.add_argument("--box-xmax", type=float, default=0.12)
    parser.add_argument("--rim-y", type=float, default=0.0)
    parser.add_argument("--box-ymax", type=float, default=0.60)
    return parser.parse_args()


def _data_array(raw: str, name: str, scope: str | None = None) -> np.ndarray:
    text = raw
    if scope is not None:
        block = re.search(
            rf"<{scope}\b[^>]*>(.*?)</{scope}>", text, flags=re.S | re.I
        )
        if not block:
            raise RuntimeError(f"missing <{scope}> block")
        text = block.group(1)
    match = re.search(
        rf"<DataArray\b(?P<a>[^>]*)\bName=['\"]{re.escape(name)}['\"]"
        rf"(?P<b>[^>]*)>(?P<body>.*?)</DataArray>",
        text,
        flags=re.S | re.I,
    )
    if not match:
        raise RuntimeError(f"missing DataArray {name!r}")
    attributes = match.group("a") + match.group("b")
    if not re.search(r"\bformat=['\"]ascii['\"]", attributes, flags=re.I):
        raise RuntimeError(f"{name!r} is not ASCII; use foamToVTK -ascii")
    return np.fromstring(match.group("body"), sep=" ", dtype=np.float64)


def read_vtu_cells(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_text(encoding="utf-8")
    points = _data_array(raw, "Points").reshape(-1, 3)
    connectivity = _data_array(raw, "connectivity").astype(np.int64)
    offsets = _data_array(raw, "offsets").astype(np.int64)
    alpha = _data_array(raw, "alpha.water", scope="CellData")
    if len(alpha) != len(offsets):
        raise RuntimeError(
            f"{path}: {len(alpha)} alpha values for {len(offsets)} cells"
        )

    counts = np.diff(np.r_[0, offsets])
    if len(counts) and np.all(counts == counts[0]):
        vertices = points[connectivity.reshape(len(offsets), int(counts[0]))]
        xmin = vertices[:, :, 0].min(axis=1)
        xmax = vertices[:, :, 0].max(axis=1)
        ymin = vertices[:, :, 1].min(axis=1)
        ymax = vertices[:, :, 1].max(axis=1)
    else:
        xmin = np.empty(len(offsets))
        xmax = np.empty(len(offsets))
        ymin = np.empty(len(offsets))
        ymax = np.empty(len(offsets))
        start = 0
        for index, end in enumerate(offsets):
            vertices = points[connectivity[start:end]]
            xmin[index] = vertices[:, 0].min()
            xmax[index] = vertices[:, 0].max()
            ymin[index] = vertices[:, 1].min()
            ymax[index] = vertices[:, 1].max()
            start = int(end)
    return {
        "alpha": alpha,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "xmid": 0.5 * (xmin + xmax),
        "ymid": 0.5 * (ymin + ymax),
        "area": (xmax - xmin) * (ymax - ymin),
    }


def find_frames(vtk_dir: Path) -> tuple[Path, list[tuple[float, Path]]]:
    series_files = sorted(vtk_dir.glob("*.vtm.series"))
    if len(series_files) != 1:
        raise RuntimeError(
            f"expected one *.vtm.series in {vtk_dir}, found {len(series_files)}"
        )
    series = series_files[0]
    payload = json.loads(series.read_text(encoding="utf-8"))
    frames: list[tuple[float, Path]] = []
    for entry in payload.get("files", []):
        stem = Path(entry["name"]).stem
        direct = vtk_dir / stem / "internal.vtu"
        matches = [direct] if direct.exists() else list((vtk_dir / stem).rglob("internal.vtu"))
        if len(matches) != 1:
            raise RuntimeError(
                f"series item {entry['name']!r}: found {len(matches)} internal.vtu files"
            )
        frames.append((float(entry["time"]), matches[0]))
    if not frames:
        raise RuntimeError(f"no VTK frames listed in {series}")
    return series, sorted(frames)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + 1.0e-10:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def measure_width(
    cells: dict[str, np.ndarray],
    external: np.ndarray,
    target_y: float,
    threshold: float,
    axis_x: float,
    box_xmin: float,
    box_xmax: float,
) -> dict[str, float | int | bool]:
    candidates = np.where(external)[0]
    if not len(candidates):
        raise RuntimeError("configured exterior contains no mesh cells")
    distances = np.abs(cells["ymid"][candidates] - target_y)
    nearest = float(distances.min())
    min_dy = float(np.min(cells["ymax"][candidates] - cells["ymin"][candidates]))
    row = candidates[distances <= nearest + max(1.0e-10, 0.05 * min_dy)]
    sampled_y = float(np.median(cells["ymid"][row]))
    wet = row[cells["alpha"][row] >= threshold]
    segments = merge_intervals(
        [(float(cells["xmin"][i]), float(cells["xmax"][i])) for i in wet]
    )
    if not segments:
        return {
            "requested_local_y_m": target_y,
            "sampled_local_y_m": sampled_y,
            "connected_segment_count": 0,
            "union_width_m": 0.0,
            "envelope_width_m": 0.0,
            "largest_segment_width_m": 0.0,
            "axis_segment_width_m": 0.0,
            "touches_left_boundary": False,
            "touches_right_boundary": False,
        }
    widths = [right - left for left, right in segments]
    axis_width = next(
        (right - left for left, right in segments if left <= axis_x <= right), 0.0
    )
    return {
        "requested_local_y_m": target_y,
        "sampled_local_y_m": sampled_y,
        "connected_segment_count": len(segments),
        "union_width_m": float(sum(widths)),
        "envelope_width_m": float(segments[-1][1] - segments[0][0]),
        "largest_segment_width_m": float(max(widths)),
        "axis_segment_width_m": float(axis_width),
        "touches_left_boundary": segments[0][0] <= box_xmin + 1.0e-8,
        "touches_right_boundary": segments[-1][1] >= box_xmax - 1.0e-8,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    args = parse_args()
    vtk_dir = args.vtk_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    series, frames = find_frames(vtk_dir)
    thresholds = sorted(set(args.thresholds))
    levels = sorted(set(args.levels_above_rim))

    height_rows: list[dict[str, object]] = []
    width_rows: list[dict[str, object]] = []
    alpha_min = float("inf")
    alpha_max = float("-inf")

    for index, (local_time, path) in enumerate(frames, start=1):
        source_time = round(local_time + args.source_time_offset, 10)
        cells = read_vtu_cells(path)
        alpha_min = min(alpha_min, float(cells["alpha"].min()))
        alpha_max = max(alpha_max, float(cells["alpha"].max()))
        external = (
            (cells["xmid"] >= args.box_xmin - 1.0e-10)
            & (cells["xmid"] <= args.box_xmax + 1.0e-10)
            & (cells["ymid"] >= args.rim_y - 1.0e-10)
            & (cells["ymid"] <= args.box_ymax + 1.0e-10)
        )
        if not np.any(external):
            raise RuntimeError(f"{path}: no cells in configured exterior")

        for threshold in thresholds:
            wet = external & (cells["alpha"] >= threshold)
            if np.any(wet):
                max_y = float(cells["ymax"][wet].max())
                height = max(max_y - args.rim_y, 0.0)
                wet_area = float(cells["area"][wet].sum())
                liquid_area = float((cells["alpha"][wet] * cells["area"][wet]).sum())
                touches_top = max_y >= args.box_ymax - 1.0e-8
                touches_side = bool(
                    np.any(cells["xmin"][wet] <= args.box_xmin + 1.0e-8)
                    or np.any(cells["xmax"][wet] >= args.box_xmax - 1.0e-8)
                )
            else:
                max_y = float("nan")
                height = 0.0
                wet_area = 0.0
                liquid_area = 0.0
                touches_top = False
                touches_side = False
            height_rows.append(
                {
                    "local_time_s": local_time,
                    "source_time_s": source_time,
                    "alpha_threshold": threshold,
                    "max_wet_local_y_m": clean_number(max_y),
                    "height_above_rim_m": height,
                    "thresholded_wet_area_m2": wet_area,
                    "liquid_area_integral_m2": liquid_area,
                    "touches_top_boundary": touches_top,
                    "touches_side_boundary": touches_side,
                }
            )
            for height_above_rim in levels:
                measured = measure_width(
                    cells,
                    external,
                    args.rim_y + height_above_rim,
                    threshold,
                    args.axis_x,
                    args.box_xmin,
                    args.box_xmax,
                )
                width_rows.append(
                    {
                        "local_time_s": local_time,
                        "source_time_s": source_time,
                        "alpha_threshold": threshold,
                        "height_above_rim_m": height_above_rim,
                        **measured,
                    }
                )
        print(
            f"[{index:02d}/{len(frames):02d}] local={local_time:.4f} s, "
            f"source={source_time:.4f} s",
            flush=True,
        )

    write_csv(output_dir / "top_plume_height_samples.csv", height_rows)
    write_csv(output_dir / "top_plume_width_samples.csv", width_rows)

    peaks: dict[str, object] = {}
    for threshold in thresholds:
        rows = [row for row in height_rows if row["alpha_threshold"] == threshold]
        peak = max(rows, key=lambda row: float(row["height_above_rim_m"]))
        wet_rows = [row for row in rows if row["max_wet_local_y_m"] is not None]
        peaks[f"alpha_ge_{threshold:g}"] = {
            "first_wet_local_time_s": (
                float(wet_rows[0]["local_time_s"]) if wet_rows else None
            ),
            "first_wet_source_time_s": (
                float(wet_rows[0]["source_time_s"]) if wet_rows else None
            ),
            "sampled_peak_local_time_s": float(peak["local_time_s"]),
            "sampled_peak_source_time_s": float(peak["source_time_s"]),
            "sampled_peak_height_above_rim_m": float(peak["height_above_rim_m"]),
            "top_boundary_contact_in_any_sample": any(
                bool(row["touches_top_boundary"]) for row in rows
            ),
            "side_boundary_contact_in_any_sample": any(
                bool(row["touches_side_boundary"]) for row in rows
            ),
        }

    metrics = {
        "case": "VW2011 Case B one-way-coupled 2D top plume",
        "source_series": str(series),
        "time_mapping": {
            "formula": "source_time_s = local_time_s + source_time_offset_s",
            "source_time_offset_s": args.source_time_offset,
            "local_times_s": [time for time, _ in frames],
            "source_times_s": [round(time + args.source_time_offset, 10) for time, _ in frames],
        },
        "geometry_local_m": {
            "rim_y": args.rim_y,
            "axis_x": args.axis_x,
            "box_xmin": args.box_xmin,
            "box_xmax": args.box_xmax,
            "box_ymax": args.box_ymax,
        },
        "alpha_thresholds": thresholds,
        "width_levels_above_rim_m": levels,
        "alpha_min_over_samples": alpha_min,
        "alpha_max_over_samples": alpha_max,
        "peaks": peaks,
        "definitions": {
            "maximum_height": (
                "Top face of the highest exterior cell whose cell-centred "
                "alpha.water meets the stated threshold."
            ),
            "width": (
                "Union of wet-cell x intervals on the mesh row nearest the "
                "requested local y level."
            ),
            "connected_segment_count": (
                "Number of disjoint threshold-wet x intervals on that row."
            ),
        },
        "evidence_status": "exploratory one-way-coupled planar supporting evidence",
        "caveats": [
            "Reported peaks are maxima over exported times, not continuous-time extrema.",
            "The source is section-reduced and one-way coupled; exterior feedback is absent.",
            "The area-equivalent planar slit does not provide a circular jet diameter.",
            "Cell-centred thresholding makes all extents mesh- and threshold-dependent.",
            "A true top/side contact flag means that the corresponding extent is domain-clipped.",
        ],
    }
    (output_dir / "top_plume_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nSampled plume peaks")
    for key, value in peaks.items():
        print(
            f"  {key}: h={value['sampled_peak_height_above_rim_m']:.4f} m, "
            f"local={value['sampled_peak_local_time_s']:.4f} s, "
            f"source={value['sampled_peak_source_time_s']:.4f} s"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
