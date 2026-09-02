#!/usr/bin/env python3
"""Quantify the resolved planar jet above the Case B tower rim.

The input is the ASCII ``internal.vtu`` series produced by
``foamToVTK -ascii -no-point-data``.  Metrics are deliberately reported for
both a liquid-core threshold (alpha.water >= 0.5) and a dilute-wet threshold
(alpha.water >= 0.01).  They are mesh- and sampling-time-dependent diagnostics,
not measurements of a circular experimental jet diameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_VTK = HERE / "VTK_PLUME_DIAGNOSTIC"
DEFAULT_OUT = HERE / "outputs" / "plume_diagnostics"

DEFAULT_RIM_Y = 0.657
DEFAULT_CROWN_Y = 0.047
DEFAULT_TOWER_LENGTH = 0.610
DEFAULT_AXIS_X = 3.516
DEFAULT_BOX_XMIN = 3.396
DEFAULT_BOX_XMAX = 3.636
DEFAULT_BOX_YMAX = 1.257


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtk-dir", type=Path, default=DEFAULT_VTK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--thresholds", type=comma_floats, default=[0.01, 0.5])
    parser.add_argument(
        "--levels-above-rim",
        type=comma_floats,
        default=[0.01, 0.05, 0.10, 0.20, 0.30, 0.50],
        help="requested horizontal sampling levels in metres above the rim",
    )
    parser.add_argument("--rim-y", type=float, default=DEFAULT_RIM_Y)
    parser.add_argument("--crown-y", type=float, default=DEFAULT_CROWN_Y)
    parser.add_argument("--tower-length", type=float, default=DEFAULT_TOWER_LENGTH)
    parser.add_argument("--axis-x", type=float, default=DEFAULT_AXIS_X)
    parser.add_argument("--box-xmin", type=float, default=DEFAULT_BOX_XMIN)
    parser.add_argument("--box-xmax", type=float, default=DEFAULT_BOX_XMAX)
    parser.add_argument("--box-ymax", type=float, default=DEFAULT_BOX_YMAX)
    return parser.parse_args()


def _data_array(raw: str, name: str, scope: str | None = None) -> np.ndarray:
    text = raw
    if scope is not None:
        match = re.search(
            rf"<{scope}\b[^>]*>(.*?)</{scope}>", text, flags=re.S | re.I
        )
        if not match:
            raise RuntimeError(f"missing <{scope}> block")
        text = match.group(1)
    match = re.search(
        rf"<DataArray\b(?P<attrs>[^>]*)\bName=['\"]{re.escape(name)}['\"]"
        rf"(?P<attrs2>[^>]*)>(?P<body>.*?)</DataArray>",
        text,
        flags=re.S | re.I,
    )
    if not match:
        raise RuntimeError(f"missing DataArray {name!r}")
    attrs = match.group("attrs") + match.group("attrs2")
    if not re.search(r"\bformat=['\"]ascii['\"]", attrs, flags=re.I):
        raise RuntimeError(
            f"DataArray {name!r} is not ASCII; rerun foamToVTK with -ascii"
        )
    return np.fromstring(match.group("body"), sep=" ", dtype=np.float64)


def read_vtu_cells(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_text(encoding="utf-8", errors="strict")
    points = _data_array(raw, "Points").reshape(-1, 3)
    connectivity = _data_array(raw, "connectivity").astype(np.int64)
    offsets = _data_array(raw, "offsets").astype(np.int64)
    alpha = _data_array(raw, "alpha.water", scope="CellData")

    if len(alpha) != len(offsets):
        raise RuntimeError(
            f"{path}: {len(alpha)} cell alpha values but {len(offsets)} cells"
        )
    counts = np.diff(np.r_[0, offsets])
    if len(counts) and np.all(counts == counts[0]):
        cell_points = points[connectivity.reshape(len(offsets), int(counts[0]))]
        xmin = cell_points[:, :, 0].min(axis=1)
        xmax = cell_points[:, :, 0].max(axis=1)
        ymin = cell_points[:, :, 1].min(axis=1)
        ymax = cell_points[:, :, 1].max(axis=1)
    else:
        xmin = np.empty(len(offsets))
        xmax = np.empty(len(offsets))
        ymin = np.empty(len(offsets))
        ymax = np.empty(len(offsets))
        start = 0
        for index, end in enumerate(offsets):
            cell_points = points[connectivity[start:end]]
            xmin[index], xmax[index] = (
                cell_points[:, 0].min(),
                cell_points[:, 0].max(),
            )
            ymin[index], ymax[index] = (
                cell_points[:, 1].min(),
                cell_points[:, 1].max(),
            )
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


def find_series(vtk_dir: Path) -> tuple[Path, list[tuple[float, Path]]]:
    candidates = sorted(vtk_dir.glob("*.vtm.series"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one *.vtm.series in {vtk_dir}, found {len(candidates)}"
        )
    series_path = candidates[0]
    payload = json.loads(series_path.read_text(encoding="utf-8"))
    frames: list[tuple[float, Path]] = []
    for item in payload.get("files", []):
        time = float(item["time"])
        stem = Path(item["name"]).stem
        direct = vtk_dir / stem / "internal.vtu"
        matches = [direct] if direct.exists() else list((vtk_dir / stem).rglob("internal.vtu"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one internal.vtu for series item {item['name']!r}, "
                f"found {len(matches)}"
            )
        frames.append((time, matches[0]))
    if not frames:
        raise RuntimeError(f"no VTK frames listed by {series_path}")
    return series_path, sorted(frames)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1] + 1.0e-10:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def width_metrics(
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
        raise RuntimeError("no cells found in configured external box")
    distance = np.abs(cells["ymid"][candidates] - target_y)
    nearest = float(distance.min())
    dy = cells["ymax"][candidates] - cells["ymin"][candidates]
    tolerance = max(1.0e-10, 0.05 * float(np.min(dy)))
    row = candidates[distance <= nearest + tolerance]
    sampled_y = float(np.median(cells["ymid"][row]))
    wet = row[cells["alpha"][row] >= threshold]
    intervals = merge_intervals(
        [(float(cells["xmin"][i]), float(cells["xmax"][i])) for i in wet]
    )
    if not intervals:
        return {
            "requested_y_m": target_y,
            "sampled_y_m": sampled_y,
            "component_count": 0,
            "union_width_m": 0.0,
            "envelope_width_m": 0.0,
            "largest_component_width_m": 0.0,
            "axis_component_width_m": 0.0,
            "touches_left_boundary": False,
            "touches_right_boundary": False,
        }
    widths = [right - left for left, right in intervals]
    axis_width = next(
        (right - left for left, right in intervals if left <= axis_x <= right), 0.0
    )
    boundary_tol = 1.0e-8
    return {
        "requested_y_m": target_y,
        "sampled_y_m": sampled_y,
        "component_count": len(intervals),
        "union_width_m": float(sum(widths)),
        "envelope_width_m": float(intervals[-1][1] - intervals[0][0]),
        "largest_component_width_m": float(max(widths)),
        "axis_component_width_m": float(axis_width),
        "touches_left_boundary": intervals[0][0] <= box_xmin + boundary_tol,
        "touches_right_boundary": intervals[-1][1] >= box_xmax - boundary_tol,
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    height_rows: list[dict[str, object]],
    width_rows: list[dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for threshold in sorted({float(row["alpha_threshold"]) for row in height_rows}):
        rows = [row for row in height_rows if row["alpha_threshold"] == threshold]
        axes[0].plot(
            [row["time_s"] for row in rows],
            [row["height_above_rim_m"] for row in rows],
            marker="o",
            ms=3,
            label=rf"$\alpha_w\geq{threshold:g}$",
        )
    axes[0].set(xlabel="Time (s)", ylabel="Resolved height above rim (m)")
    axes[0].legend(frameon=False)

    core_threshold = min(
        {float(row["alpha_threshold"]) for row in width_rows},
        key=lambda value: abs(value - 0.5),
    )
    core_rows = [row for row in width_rows if row["alpha_threshold"] == core_threshold]
    levels = sorted({float(row["height_above_rim_m"]) for row in core_rows})
    for level in levels:
        rows = [row for row in core_rows if row["height_above_rim_m"] == level]
        axes[1].plot(
            [row["time_s"] for row in rows],
            [row["union_width_m"] for row in rows],
            marker="o",
            ms=3,
            label=f"z={level:.2f} m",
        )
    axes[1].set(
        xlabel="Time (s)",
        ylabel=rf"Wet width (m), $\alpha_w\geq{core_threshold:g}$",
    )
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vtk_dir = args.vtk_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    series_path, frames = find_series(vtk_dir)

    thresholds = sorted(set(args.thresholds))
    levels = sorted(set(args.levels_above_rim))
    height_rows: list[dict[str, object]] = []
    width_rows: list[dict[str, object]] = []
    alpha_bounds: list[tuple[float, float]] = []

    for index, (time, path) in enumerate(frames, start=1):
        cells = read_vtu_cells(path)
        alpha_bounds.append((float(cells["alpha"].min()), float(cells["alpha"].max())))
        external = (
            (cells["xmid"] >= args.box_xmin - 1.0e-10)
            & (cells["xmid"] <= args.box_xmax + 1.0e-10)
            & (cells["ymid"] >= args.rim_y - 1.0e-10)
            & (cells["ymid"] <= args.box_ymax + 1.0e-10)
        )
        if not np.any(external):
            raise RuntimeError(f"{path}: no cells inside configured external box")

        for threshold in thresholds:
            wet = external & (cells["alpha"] >= threshold)
            if np.any(wet):
                max_y = float(cells["ymax"][wet].max())
                wet_area = float(cells["area"][wet].sum())
                liquid_area = float((cells["alpha"][wet] * cells["area"][wet]).sum())
                touches_top = max_y >= args.box_ymax - 1.0e-8
                touches_side = bool(
                    np.any(cells["xmin"][wet] <= args.box_xmin + 1.0e-8)
                    or np.any(cells["xmax"][wet] >= args.box_xmax - 1.0e-8)
                )
            else:
                max_y = float("nan")
                wet_area = 0.0
                liquid_area = 0.0
                touches_top = False
                touches_side = False
            height = max(max_y - args.rim_y, 0.0) if math.isfinite(max_y) else 0.0
            height_rows.append(
                {
                    "time_s": time,
                    "alpha_threshold": threshold,
                    "max_wet_y_m": finite_or_none(max_y),
                    "height_above_rim_m": height,
                    "max_wet_Ystar": (
                        finite_or_none((max_y - args.crown_y) / args.tower_length)
                        if math.isfinite(max_y)
                        else None
                    ),
                    "thresholded_wet_area_m2": wet_area,
                    "liquid_area_integral_m2": liquid_area,
                    "touches_top_boundary": touches_top,
                    "touches_side_boundary": touches_side,
                }
            )
            for height_above_rim in levels:
                metrics = width_metrics(
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
                        "time_s": time,
                        "alpha_threshold": threshold,
                        "height_above_rim_m": height_above_rim,
                        **metrics,
                    }
                )
        print(f"[{index:02d}/{len(frames):02d}] t={time:g} s  {path.name}", flush=True)

    write_csv(output_dir / "external_plume_height_samples.csv", height_rows)
    write_csv(output_dir / "external_plume_width_samples.csv", width_rows)
    write_plot(output_dir / "external_plume_diagnostics.png", height_rows, width_rows)

    peak_by_threshold: dict[str, object] = {}
    for threshold in thresholds:
        rows = [row for row in height_rows if row["alpha_threshold"] == threshold]
        peak = max(rows, key=lambda row: float(row["height_above_rim_m"]))
        wet_times = [float(row["time_s"]) for row in rows if row["max_wet_y_m"] is not None]
        peak_by_threshold[f"alpha_ge_{threshold:g}"] = {
            "first_sample_with_water_above_rim_s": min(wet_times) if wet_times else None,
            "sampled_peak_time_s": float(peak["time_s"]),
            "sampled_peak_height_above_rim_m": float(peak["height_above_rim_m"]),
            "sampled_peak_Ystar": peak["max_wet_Ystar"],
            "top_boundary_contact_in_any_sample": any(
                bool(row["touches_top_boundary"]) for row in rows
            ),
            "side_boundary_contact_in_any_sample": any(
                bool(row["touches_side_boundary"]) for row in rows
            ),
        }

    metrics = {
        "case": "VW2011 Test 1 Case B external-plume 2D surrogate",
        "source_series": str(series_path),
        "sample_times_s": [time for time, _ in frames],
        "alpha_thresholds": thresholds,
        "geometry_m": {
            "tower_rim_y": args.rim_y,
            "tower_crown_y": args.crown_y,
            "tower_length": args.tower_length,
            "tower_axis_x": args.axis_x,
            "external_box_xmin": args.box_xmin,
            "external_box_xmax": args.box_xmax,
            "external_box_ymax": args.box_ymax,
        },
        "requested_width_levels_above_rim_m": levels,
        "alpha_min_over_samples": min(value[0] for value in alpha_bounds),
        "alpha_max_over_samples": max(value[1] for value in alpha_bounds),
        "peaks": peak_by_threshold,
        "definitions": {
            "height": (
                "Top face of the highest external-box cell whose cell-centred "
                "alpha.water meets the threshold."
            ),
            "width": (
                "Union of x-intervals of threshold-wet cells on the mesh row "
                "nearest each requested y level."
            ),
            "axis_component_width": (
                "Width of the connected wet interval containing x=axis_x; zero "
                "when the wet set is detached from the centreline."
            ),
        },
        "evidence_status": "exploratory supporting 2D evidence",
        "caveats": [
            "The sampled peak is a maximum over exported times, not a continuous-time maximum.",
            "The tower is an area-equivalent planar slit; lateral widths are not circular jet diameters.",
            "Cell-centred thresholding makes height and width mesh- and threshold-dependent.",
            "Any top or side boundary contact means the corresponding extent is clipped by the domain.",
        ],
    }
    (output_dir / "external_plume_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print("\nSampled peaks")
    for key, value in peak_by_threshold.items():
        print(
            f"  {key}: {value['sampled_peak_height_above_rim_m']:.4f} m "
            f"at t={value['sampled_peak_time_s']:.3f} s; "
            f"top-contact={value['top_boundary_contact_in_any_sample']}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
