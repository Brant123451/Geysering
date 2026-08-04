"""Audit Case-A horizontal 1-D and 2-D fields with one explicit metric basis.

This script is diagnostic only.  It fixes three ambiguities that otherwise
change the answer appreciably:

* all requested times are read from the complete ``VTK_CASEA_HTML`` series;
  the older junction-baseline extractor intentionally stops at 10.75 s;
* the 1-D liquid variable is a *circular-pipe area fraction*, so interface
  height is obtained by circular-segment inversion rather than ``D*alpha``;
* gas inventory is compared by preserving section-averaged void fraction and
  mapping both models to the same physical circular area.

Interface statistics are reported on a common 4 mm x-grid.  ``resolved``
statistics apply the same 25 mm quadratic Savitzky--Golay pass used by the
existing 2-D junction baseline.  A resolved crest is counted only if its
prominence is at least 2 mm and it is separated from another crest by at
least 48 mm; this rejects cell-scale VOF stair stepping.  Volume and the
alpha_l < 0.5 length are integrated conservatively from source cell averages,
not from the interpolated plotting profile.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, savgol_filter


HERE = Path(__file__).resolve().parent
CASE = HERE.parent

D = 0.094
DT = 0.0571
L_TUNNEL = 0.546 + 2.970 + 0.490
TOWER_X = 3.516
TOWER_LEFT = TOWER_X - 0.5 * DT
PIPE_AREA = math.pi * D * D / 4.0

COMMON_DX = 0.004
RESOLVE_LENGTH = 0.025
TREND_LENGTH = 0.300
PEAK_PROMINENCE = 0.002
PEAK_SEPARATION = 0.048

REGIONS = {
    "upstream_to_tower": (0.0, TOWER_LEFT),
    "junction_wave": (2.45, TOWER_LEFT),
}

INVENTORY_REGIONS = {
    "upstream_to_tower": (0.0, TOWER_LEFT),
    "tower_to_closed_end": (TOWER_LEFT, L_TUNNEL),
    "full_horizontal_pipe": (0.0, L_TUNNEL),
}


def _array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Expected ASCII foamToVTK output")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def _read_vtu(path: Path, read_geometry: bool) -> dict[str, np.ndarray | float]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    alpha_node = root.find(".//CellData/DataArray[@Name='alpha.water']")
    if time_node is None or alpha_node is None:
        raise ValueError(f"Missing TimeValue or alpha.water in {path}")
    result: dict[str, np.ndarray | float] = {
        "time": float((time_node.text or "0").strip()),
        "alpha": np.clip(_array(alpha_node), 0.0, 1.0),
    }
    if not read_geometry:
        return result

    points_node = root.find(".//Points/DataArray")
    if points_node is None:
        raise ValueError(f"Missing points in {path}")
    points = _array(points_node, 3)
    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = _array(arrays["connectivity"]).astype(int)
    offsets = _array(arrays["offsets"]).astype(int)
    cells = [connectivity[i:j] for i, j in zip(np.r_[0, offsets[:-1]], offsets)]
    centres = np.empty((len(cells), 2))
    areas = np.empty(len(cells))
    for i, cell in enumerate(cells):
        xy = np.unique(points[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        polygon = xy[order]
        x = polygon[:, 0]
        y = polygon[:, 1]
        centres[i] = centre
        areas[i] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    result.update(centres=centres, areas=areas)
    return result


def _all_vtu_paths(vtk_root: Path) -> list[tuple[float, Path]]:
    found: list[tuple[float, Path]] = []
    for path in vtk_root.glob("*/internal.vtu"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.read(512)
        match = re.search(r"\btime='([^']+)'", header)
        if match is None:
            raise ValueError(f"Cannot read time from VTU header: {path}")
        found.append((float(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise RuntimeError(f"No VTU files found under {vtk_root}")
    return found


def _column_profile(
    alpha: np.ndarray,
    centres: np.ndarray,
    areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pipe = (
        (centres[:, 0] >= -1.0e-8)
        & (centres[:, 0] <= L_TUNNEL + 1.0e-8)
        & (centres[:, 1] >= -0.5 * D - 1.0e-8)
        & (centres[:, 1] <= 0.5 * D + 1.0e-8)
    )
    x_values = np.unique(np.round(centres[pipe, 0], 8))
    liquid_fraction = np.empty_like(x_values)
    for i, x in enumerate(x_values):
        use = pipe & np.isclose(centres[:, 0], x, atol=2.0e-7)
        liquid_fraction[i] = np.average(alpha[use], weights=areas[use])
    return x_values, np.clip(liquid_fraction, 0.0, 1.0)


def _circular_depth_fraction(area_fraction: np.ndarray) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, 20001)
    area = (theta - np.sin(theta)) / (2.0 * math.pi)
    depth = 0.5 * (1.0 - np.cos(0.5 * theta))
    return np.interp(np.clip(area_fraction, 0.0, 1.0), area, depth)


def _odd_window(target_length: float, dx: float, n: int, minimum: int) -> int:
    raw = max(minimum, int(round(target_length / dx)))
    if raw % 2 == 0:
        raw += 1
    maximum = n if n % 2 else n - 1
    return min(raw, maximum)


def _edge_hold_pchip(x: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    clipped = np.clip(target, x[0], x[-1])
    return PchipInterpolator(x, values, extrapolate=False)(clipped)


def _cell_overlap_integral(
    x: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
) -> float:
    edges = np.empty(len(x) + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    overlap = np.maximum(0.0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    return float(np.sum(values * overlap))


def _profile_metrics(x: np.ndarray, gas_thickness: np.ndarray) -> dict[str, object]:
    dx = float(np.median(np.diff(x)))
    resolve_window = _odd_window(RESOLVE_LENGTH, dx, len(x), minimum=5)
    trend_window = _odd_window(TREND_LENGTH, dx, len(x), minimum=9)
    resolved = savgol_filter(gas_thickness, resolve_window, 2, mode="interp")
    trend = savgol_filter(resolved, trend_window, 2, mode="interp")
    residual = resolved - trend
    separation_samples = max(1, int(round(PEAK_SEPARATION / dx)))
    peaks, properties = find_peaks(
        resolved,
        prominence=PEAK_PROMINENCE,
        distance=separation_samples,
    )

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_gas_thickness_m": float(np.mean(values)),
            "p95_gas_thickness_m": float(np.percentile(values, 95)),
            "maximum_gas_thickness_m": float(np.max(values)),
            "total_variation_m": float(np.sum(np.abs(np.diff(values)))),
        }

    return {
        "common_dx_m": dx,
        "raw_common_grid": summary(gas_thickness),
        "resolved_25mm": {
            **summary(resolved),
            "crest_count_prominence_ge_2mm": int(len(peaks)),
            "crest_x_m": [float(value) for value in x[peaks]],
            "crest_prominence_m": [float(value) for value in properties["prominences"]],
        },
        "resolved_minus_300mm_trend": {
            "rms_m": float(np.sqrt(np.mean(residual * residual))),
            "peak_to_peak_m": float(np.ptp(residual)),
            "total_variation_m": float(np.sum(np.abs(np.diff(residual)))),
        },
    }


def _region_metrics(
    source_x: np.ndarray,
    source_liquid_fraction: np.ndarray,
    source_gas_thickness: np.ndarray,
    lo: float,
    hi: float,
) -> dict[str, object]:
    x = np.arange(lo, hi + 0.25 * COMMON_DX, COMMON_DX)
    x[-1] = min(x[-1], hi)
    gas_thickness = _edge_hold_pchip(source_x, source_gas_thickness, x)
    gas_volume = PIPE_AREA * _cell_overlap_integral(
        source_x, 1.0 - source_liquid_fraction, lo, hi
    )
    gas_rich_length = _cell_overlap_integral(
        source_x, (source_liquid_fraction < 0.5).astype(float), lo, hi
    )
    return {
        "x_interval_m": [lo, hi],
        "gas_volume_m3_circular_area_mapping": gas_volume,
        "gas_rich_alpha_l_lt_0p5_length_m": gas_rich_length,
        "interface": _profile_metrics(x, gas_thickness),
    }


def _inventory_metrics(
    source_x: np.ndarray,
    source_liquid_fraction: np.ndarray,
    lo: float,
    hi: float,
) -> dict[str, float | list[float]]:
    return {
        "x_interval_m": [lo, hi],
        "gas_volume_m3_circular_area_mapping": PIPE_AREA
        * _cell_overlap_integral(source_x, 1.0 - source_liquid_fraction, lo, hi),
        "gas_rich_alpha_l_lt_0p5_length_m": _cell_overlap_integral(
            source_x, (source_liquid_fraction < 0.5).astype(float), lo, hi
        ),
    }


def _interpolate_time(
    time: np.ndarray,
    values: np.ndarray,
    target: float,
) -> tuple[np.ndarray, list[float]]:
    """Linearly interpolate stored cell averages to an exact comparison time."""
    if target <= time[0]:
        return values[0].copy(), [float(time[0]), float(time[0])]
    if target >= time[-1]:
        return values[-1].copy(), [float(time[-1]), float(time[-1])]
    right = int(np.searchsorted(time, target, side="left"))
    if abs(time[right] - target) <= 1.0e-10:
        return values[right].copy(), [float(time[right]), float(time[right])]
    left = right - 1
    fraction = float((target - time[left]) / (time[right] - time[left]))
    interpolated = (1.0 - fraction) * values[left] + fraction * values[right]
    return interpolated, [float(time[left]), float(time[right])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fields", type=Path, help="1-D vertical_fields_*.npz")
    parser.add_argument(
        "--vtk-root",
        type=Path,
        default=CASE / "openfoam/2d/VTK_CASEA_HTML",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[9.0, 9.35, 10.0, 11.0, 12.0, 13.0],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fields = np.load(args.fields)
    time_1d = np.asarray(fields["time"], dtype=float)
    alpha_1d = np.clip(np.asarray(fields["horizontal_alpha_l"], dtype=float), 0.0, 1.0)
    nx = alpha_1d.shape[1]
    dx_1d = L_TUNNEL / nx
    x_1d = (np.arange(nx) + 0.5) * dx_1d

    series = _all_vtu_paths(args.vtk_root)
    first = _read_vtu(series[0][1], read_geometry=True)
    centres = np.asarray(first["centres"])
    areas = np.asarray(first["areas"])

    snapshots: dict[str, object] = {}
    for target in args.times:
        time_2d, path_2d = min(series, key=lambda item: abs(item[0] - target))
        data_2d = _read_vtu(path_2d, read_geometry=False)

        x_2d, liquid_fraction_2d = _column_profile(
            np.asarray(data_2d["alpha"]), centres, areas
        )
        # The 2-D slice is planar: section average equals water-depth fraction.
        gas_thickness_2d = D * (1.0 - liquid_fraction_2d)
        # The 1-D state is circular wetted-area fraction and needs inversion.
        liquid_fraction_1d, bracket_1d = _interpolate_time(time_1d, alpha_1d, target)
        gas_thickness_1d = D * (
            1.0 - _circular_depth_fraction(liquid_fraction_1d)
        )

        by_region: dict[str, object] = {}
        for name, (lo, hi) in REGIONS.items():
            by_region[name] = {
                "2d": _region_metrics(
                    x_2d, liquid_fraction_2d, gas_thickness_2d, lo, hi
                ),
                "1d": _region_metrics(
                    x_1d, liquid_fraction_1d, gas_thickness_1d, lo, hi
                ),
            }
        inventories: dict[str, object] = {}
        for name, (lo, hi) in INVENTORY_REGIONS.items():
            inventories[name] = {
                "2d": _inventory_metrics(x_2d, liquid_fraction_2d, lo, hi),
                "1d": _inventory_metrics(x_1d, liquid_fraction_1d, lo, hi),
            }
        snapshots[f"{target:.2f}"] = {
            "requested_time_s": target,
            "interpolated_1d_time_s": target,
            "source_1d_time_bracket_s": bracket_1d,
            "selected_2d_time_s": float(time_2d),
            "horizontal_gas_inventory": inventories,
            "regions": by_region,
        }

    result = {
        "case": "Vasconcelos and Wright (2011), Case A",
        "fields_1d": str(args.fields.resolve()),
        "vtk_2d": str(args.vtk_root.resolve()),
        "definitions": {
            "2d_interface": "D times column-averaged alpha.water; planar slice",
            "1d_interface": "circular-segment depth reconstructed from wetted area fraction",
            "gas_volume": (
                "integral of section-averaged void fraction times the physical circular "
                "pipe area; source cell averages and exact interval overlaps"
            ),
            "full_horizontal_2d_inventory": (
                "All 2-D columns with y inside the horizontal bore are included from x=0 "
                "to 4.006 m; the T-opening columns are part of the shared junction cavity."
            ),
            "resolved_profile": "quadratic Savitzky-Golay, 0.025 m physical window",
            "crest_count": (
                "resolved gas-thickness maxima with >=0.002 m prominence and >=0.048 m separation"
            ),
            "important_time_note": (
                "The complete VTU directory is scanned. caseA_extract_2d_junction_baseline.py "
                "uses READ_END=10.75 s and must not be used to select a 13 s frame."
            ),
        },
        "snapshots": snapshots,
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
