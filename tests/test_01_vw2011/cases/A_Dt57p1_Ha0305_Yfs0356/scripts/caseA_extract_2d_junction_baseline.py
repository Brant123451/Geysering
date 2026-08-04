"""Extract a quantitative Case-A junction baseline from the 2-D VTU series.

This diagnostic intentionally does not alter either solver.  It maps the
planar OpenFOAM tower fields to the physical circular tower by preserving the
section-averaged phase fraction.  The resulting liquid volume and its time
derivative are therefore proxies for comparison with the circular 1-D model,
not a claim that the planar and circular cross-sections are identical.

The horizontal interface trace is the section-integrated water depth in the
left-of-tower pipe.  A short spatial Savitzky--Golay pass removes cell-scale
VOF stair-stepping; a 0.30 m Savitzky--Golay trend is then subtracted to
separate the local wave packet from the slowly varying pocket envelope.
"""

from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
VTK_ROOT = CASE / "openfoam/2d/VTK_CASEA_HTML"
LEVELS_CSV = CASE / "openfoam/2d/outputs/openfoam_2d_levels.csv"
OUT = CASE / "outputs/caseA_openfoam2d_junction_wave_baseline_7p5_10p5s.json"

D = 0.094
DT = 0.0571
TOWER_X = 3.516
TOWER_LEFT = TOWER_X - 0.5 * DT
TOWER_RIGHT = TOWER_X + 0.5 * DT
CROWN_Y = 0.5 * D
TOWER_HEIGHT = 0.610
TOWER_TOP = CROWN_Y + TOWER_HEIGHT
AT = math.pi * DT * DT / 4.0

READ_START = 7.25
READ_END = 10.75
REPORT_START = 7.50
REPORT_END = 10.50
WAVE_X_MIN = 2.45
WAVE_X_MAX = TOWER_LEFT


def _array(node: ET.Element, ncomp: int = 1) -> np.ndarray:
    if node.attrib.get("format") != "ascii":
        raise ValueError("Expected ASCII foamToVTK output")
    values = np.fromstring(node.text or "", sep=" ")
    return values.reshape(-1, ncomp) if ncomp > 1 else values


def _read_vtu(path: Path, read_geometry: bool) -> dict[str, np.ndarray | float]:
    root = ET.parse(path).getroot()
    time_node = root.find(".//FieldData/DataArray[@Name='TimeValue']")
    if time_node is None:
        raise ValueError(f"Missing TimeValue in {path}")
    cell_data = {
        node.attrib.get("Name"): node for node in root.findall(".//CellData/DataArray")
    }
    out: dict[str, np.ndarray | float] = {
        "time": float((time_node.text or "0").strip()),
        "alpha": np.clip(_array(cell_data["alpha.water"]), 0.0, 1.0),
    }
    if "U" in cell_data:
        out["velocity"] = _array(cell_data["U"], 3)
    if not read_geometry:
        return out

    points = _array(root.find(".//Points/DataArray"), 3)
    arrays = {node.attrib.get("Name"): node for node in root.findall(".//Cells/DataArray")}
    connectivity = _array(arrays["connectivity"]).astype(int)
    offsets = _array(arrays["offsets"]).astype(int)
    cells = [connectivity[i:j] for i, j in zip(np.r_[0, offsets[:-1]], offsets)]
    centres = np.empty((len(cells), 2))
    areas = np.empty(len(cells))
    widths = np.empty(len(cells))
    heights = np.empty(len(cells))
    for i, cell in enumerate(cells):
        xy = np.unique(points[cell, :2], axis=0)
        centre = xy.mean(axis=0)
        order = np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
        poly = xy[order]
        x = poly[:, 0]
        y = poly[:, 1]
        centres[i] = centre
        areas[i] = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        widths[i] = float(np.max(x) - np.min(x))
        heights[i] = float(np.max(y) - np.min(y))
    out.update(centres=centres, areas=areas, widths=widths, heights=heights)
    return out


def _series_paths() -> list[tuple[float, Path]]:
    # The series manifest was regenerated after a resumed run and omits part
    # of the earlier time range, while the per-time VTU directories are
    # complete.  Read the authoritative TimeValue/comment from every VTU.
    found: list[tuple[float, Path]] = []
    for path in VTK_ROOT.glob("*/internal.vtu"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.read(512)
        match = re.search(r"\btime='([^']+)'", header)
        if match is None:
            raise ValueError(f"Cannot read time from VTU header: {path}")
        time = float(match.group(1))
        if READ_START - 1e-10 <= time <= READ_END + 1e-10:
            found.append((time, path))
    found.sort(key=lambda item: item[0])
    return found


def _odd_window(target_length: float, spacing: float, n: int, minimum: int = 5) -> int:
    raw = max(minimum, int(round(target_length / spacing)))
    if raw % 2 == 0:
        raw += 1
    max_window = n if n % 2 == 1 else n - 1
    return min(raw, max_window)


def _main_wet_segment(profile: np.ndarray, threshold: float = 0.5) -> tuple[int, int] | None:
    wet = profile >= threshold
    padded = np.r_[False, wet, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    if starts.size == 0:
        return None
    lengths = stops - starts + 1
    valid = np.flatnonzero(lengths >= 3)
    if valid.size == 0:
        return None
    longest = valid[lengths[valid] == np.max(lengths[valid])]
    selected = int(longest[np.argmax(stops[longest])])
    return int(starts[selected]), int(stops[selected])


def _crossing(x0: float, x1: float, a0: float, a1: float, target: float = 0.5) -> float:
    if abs(a1 - a0) < 1e-12:
        return 0.5 * (x0 + x1)
    f = float(np.clip((target - a0) / (a1 - a0), 0.0, 1.0))
    return x0 + f * (x1 - x0)


def _riser_surface_height(
    alpha: np.ndarray,
    centres: np.ndarray,
    areas: np.ndarray,
    tower_mask: np.ndarray,
) -> float:
    y_values = np.unique(np.round(centres[tower_mask, 1], 8))
    profile = np.empty_like(y_values)
    for i, y in enumerate(y_values):
        use = tower_mask & np.isclose(centres[:, 1], y, atol=2e-7)
        profile[i] = np.average(alpha[use], weights=areas[use])
    segment = _main_wet_segment(profile)
    if segment is None:
        return float("nan")
    _, last = segment
    if last == len(profile) - 1:
        top = TOWER_TOP
    else:
        top = _crossing(y_values[last], y_values[last + 1], profile[last], profile[last + 1])
    return max(float(top - CROWN_Y), 0.0)


def _horizontal_profile(
    alpha: np.ndarray,
    centres: np.ndarray,
    areas: np.ndarray,
    pipe_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.unique(np.round(centres[pipe_mask, 0], 8))
    depth = np.empty_like(x_values)
    for i, x in enumerate(x_values):
        use = pipe_mask & np.isclose(centres[:, 0], x, atol=2e-7)
        depth[i] = D * np.average(alpha[use], weights=areas[use])
    return x_values, depth


def _active_component(x: np.ndarray, envelope: np.ndarray, threshold: float) -> dict[str, float | list[float]]:
    active = envelope >= threshold
    padded = np.r_[False, active, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1) - 1
    if starts.size == 0:
        return {"largest_length_m": 0.0, "total_length_m": 0.0, "largest_bounds_m": []}
    dx = float(np.median(np.diff(x)))
    lengths = (stops - starts + 1) * dx
    selected = int(np.argmax(lengths))
    return {
        "largest_length_m": float(lengths[selected]),
        "total_length_m": float(np.sum(lengths)),
        "largest_bounds_m": [
            float(x[starts[selected]] - 0.5 * dx),
            float(x[stops[selected]] + 0.5 * dx),
        ],
    }


def _wave_metrics(x: np.ndarray, depth: np.ndarray) -> dict[str, float | list[float]]:
    dx = float(np.median(np.diff(x)))
    small_window = _odd_window(0.025, dx, len(x), minimum=5)
    trend_window = _odd_window(0.30, dx, len(x), minimum=9)
    envelope_window = _odd_window(0.08, dx, len(x), minimum=5)
    resolved = savgol_filter(depth, small_window, 2, mode="interp")
    trend = savgol_filter(resolved, trend_window, 2, mode="interp")
    residual = resolved - trend
    kernel = np.ones(envelope_window) / envelope_window
    envelope = np.sqrt(np.convolve(residual * residual, kernel, mode="same"))
    active_2mm = _active_component(x, envelope, 0.002)
    active_halfmax = _active_component(x, envelope, 0.5 * float(np.max(envelope)))
    return {
        "x_min_m": float(x[0]),
        "x_max_m": float(x[-1]),
        "dx_m": dx,
        "profile_a90_m": float(np.percentile(resolved, 95) - np.percentile(resolved, 5)),
        "residual_a90_m": float(np.percentile(residual, 95) - np.percentile(residual, 5)),
        "residual_peak_to_peak_m": float(np.ptp(residual)),
        "residual_rms_m": float(np.sqrt(np.mean(residual * residual))),
        "envelope_max_m": float(np.max(envelope)),
        "active_fixed_2mm": active_2mm,
        "active_half_max": active_halfmax,
    }


def _savgol_time(values: np.ndarray, times: np.ndarray, derivative: int = 0) -> np.ndarray:
    dt = float(np.median(np.diff(times)))
    window = _odd_window(0.25, dt, len(values), minimum=5)
    return savgol_filter(values, window, 3, deriv=derivative, delta=dt, mode="interp")


def _savgol_time_width(
    values: np.ndarray, times: np.ndarray, width_s: float, derivative: int = 0
) -> np.ndarray:
    dt = float(np.median(np.diff(times)))
    window = _odd_window(width_s, dt, len(values), minimum=5)
    return savgol_filter(values, window, 3, deriv=derivative, delta=dt, mode="interp")


def _reversal_times(times: np.ndarray, flow: np.ndarray, threshold: float) -> list[float]:
    state = 0
    reversals: list[float] = []
    for time, value in zip(times, flow):
        sign = 1 if value > threshold else -1 if value < -threshold else 0
        if sign == 0:
            continue
        if state and sign != state:
            reversals.append(float(time))
        state = sign
    return reversals


def _window_summary(times: np.ndarray, flow: np.ndarray, lo: float, hi: float) -> dict[str, object]:
    use = (times >= lo - 1e-10) & (times <= hi + 1e-10)
    q = flow[use]
    t = times[use]
    threshold = 5.0e-6  # 0.005 L/s; rejects near-zero VOF jitter.
    return {
        "time_window_s": [lo, hi],
        "minimum_m3_s": float(np.min(q)),
        "maximum_m3_s": float(np.max(q)),
        "minimum_L_s": float(np.min(q) * 1000.0),
        "maximum_L_s": float(np.max(q) * 1000.0),
        "rms_m3_s": float(np.sqrt(np.mean(q * q))),
        "rms_L_s": float(np.sqrt(np.mean(q * q)) * 1000.0),
        "mean_m3_s": float(np.mean(q)),
        "mean_L_s": float(np.mean(q) * 1000.0),
        "reversal_hysteresis_L_s": threshold * 1000.0,
        "reversal_times_s": _reversal_times(t, q, threshold),
        "reversal_count": len(_reversal_times(t, q, threshold)),
        "reversal_count_sensitivity": {
            "zero_threshold": len(_reversal_times(t, q, 0.0)),
            "0.002_L_s": len(_reversal_times(t, q, 2.0e-6)),
            "0.010_L_s": len(_reversal_times(t, q, 1.0e-5)),
        },
    }


def _level_lift() -> dict[str, float | list[float] | str]:
    with LEVELS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    time = np.asarray([float(row["time_s"]) for row in rows])
    yfs = np.asarray([float(row["Yfs_star"]) for row in rows]) * TOWER_HEIGHT
    baseline = (time >= 6.0) & (time <= 6.5) & np.isfinite(yfs)
    event = (time >= 6.5) & (time <= 7.6) & np.isfinite(yfs)
    base = float(np.median(yfs[baseline]))
    peak_index = np.flatnonzero(event)[int(np.argmax(yfs[event]))]
    return {
        "source": "existing centreline probe post-processing, 0.005 s samples",
        "baseline_window_s": [6.0, 6.5],
        "baseline_median_height_above_crown_m": base,
        "peak_height_above_crown_m": float(yfs[peak_index]),
        "peak_time_s": float(time[peak_index]),
        "peak_lift_from_baseline_m": float(yfs[peak_index] - base),
        "last_finite_level_time_s": float(time[np.flatnonzero(np.isfinite(yfs))[-1]]),
    }


def main() -> None:
    datasets = _series_paths()
    if not datasets:
        raise RuntimeError("No VTU datasets selected")

    first = _read_vtu(datasets[0][1], read_geometry=True)
    centres = np.asarray(first["centres"])
    areas = np.asarray(first["areas"])
    heights = np.asarray(first["heights"])
    tower_mask = (
        (centres[:, 0] >= TOWER_LEFT - 1e-8)
        & (centres[:, 0] <= TOWER_RIGHT + 1e-8)
        & (centres[:, 1] >= CROWN_Y - 1e-8)
        & (centres[:, 1] <= TOWER_TOP + 1e-8)
    )
    tower_y = centres[tower_mask, 1]
    base_y = float(np.min(tower_y))
    top_y = float(np.max(tower_y))
    row_tol = 0.25 * float(np.min(heights[tower_mask]))
    base_mask = tower_mask & np.isclose(centres[:, 1], base_y, atol=row_tol)
    top_mask = tower_mask & np.isclose(centres[:, 1], top_y, atol=row_tol)
    pipe_mask = (
        (centres[:, 0] >= WAVE_X_MIN - 1e-8)
        & (centres[:, 0] <= WAVE_X_MAX + 1e-8)
        & (centres[:, 1] >= -0.5 * D - 1e-8)
        & (centres[:, 1] <= 0.5 * D + 1e-8)
    )

    records: list[dict[str, object]] = []
    for index, (series_time, path) in enumerate(datasets):
        data = first if index == 0 else _read_vtu(path, read_geometry=False)
        time = float(data["time"])
        if abs(time - series_time) > 1e-5:
            raise RuntimeError(f"Series/VTU time mismatch at {path}: {series_time} vs {time}")
        alpha = np.asarray(data["alpha"])

        # Section-averaged planar inventory mapped to the physical circular area.
        # VTK_CASEA_HTML intentionally contains only alpha.water, so the base
        # liquid flux is recovered from dV/dt below.  This is valid while the
        # open-top row remains dry, which is checked and reported explicitly.
        base_water_fraction = float(np.average(alpha[base_mask], weights=areas[base_mask]))
        top_water_fraction = float(np.average(alpha[top_mask], weights=areas[top_mask]))
        equivalent_liquid_height = float(np.sum(alpha[tower_mask] * areas[tower_mask]) / DT)
        x, depth = _horizontal_profile(alpha, centres, areas, pipe_mask)
        records.append(
            {
                "time_s": time,
                "riser_equivalent_liquid_height_m": equivalent_liquid_height,
                "riser_equivalent_liquid_volume_m3": AT * equivalent_liquid_height,
                "base_row_water_fraction": base_water_fraction,
                "top_row_water_fraction": top_water_fraction,
                "primary_wet_segment_top_above_crown_m": _riser_surface_height(alpha, centres, areas, tower_mask),
                "horizontal_wave": _wave_metrics(x, depth),
            }
        )
        print(f"Read {index + 1:02d}/{len(datasets)}: {time:.2f} s")

    times = np.asarray([float(row["time_s"]) for row in records])
    volumes = np.asarray([float(row["riser_equivalent_liquid_volume_m3"]) for row in records])
    q_volume = _savgol_time(volumes, times, derivative=1)
    volume_smooth = _savgol_time(volumes, times)

    for i, row in enumerate(records):
        row.update(
            {
                "riser_equivalent_liquid_volume_smoothed_m3": float(volume_smooth[i]),
                "volume_derivative_flux_m3_s": float(q_volume[i]),
            }
        )

    report = (times >= REPORT_START - 1e-10) & (times <= REPORT_END + 1e-10)
    wave_target_times = [8.0, 9.0, 9.35, 10.0, 10.5]
    wave_snapshots: dict[str, object] = {}
    for target in wave_target_times:
        i = int(np.argmin(np.abs(times - target)))
        wave_snapshots[f"{times[i]:.2f}"] = records[i]["horizontal_wave"]
    residual_a90 = np.asarray(
        [float(row["horizontal_wave"]["residual_a90_m"]) for row in records]
    )
    active_length = np.asarray(
        [float(row["horizontal_wave"]["active_fixed_2mm"]["largest_length_m"]) for row in records]
    )

    smoothing_sensitivity: dict[str, object] = {}
    for width in (0.25, 0.45, 0.75):
        q_width = _savgol_time_width(volumes, times, width, derivative=1)
        smoothing_sensitivity[f"{width:.2f}_s"] = {
            "8.0_to_10.0_s": _window_summary(times, q_width, 8.0, 10.0),
            "7.5_to_10.5_s": _window_summary(times, q_width, 7.5, 10.5),
        }

    direct_cross_check: dict[str, object] = {
        "available_time_s": 9.35,
        "source": "VTK_KH_935 export containing alpha.water and U",
    }
    kh_paths = list((CASE / "openfoam/2d/VTK_KH_935").glob("*/internal.vtu"))
    if kh_paths:
        kh = _read_vtu(kh_paths[0], read_geometry=False)
        kh_alpha = np.asarray(kh["alpha"])
        kh_velocity = np.asarray(kh["velocity"])
        superficial_vertical_velocity = float(
            np.average(kh_alpha[base_mask] * kh_velocity[base_mask, 1], weights=areas[base_mask])
        )
        kh_q = AT * superficial_vertical_velocity
        nearest = int(np.argmin(np.abs(times - float(kh["time"]))))
        direct_cross_check.update(
            {
                "time_s": float(kh["time"]),
                "first_tower_row_direct_flux_m3_s": kh_q,
                "first_tower_row_direct_flux_L_s": kh_q * 1000.0,
                "inventory_derivative_flux_m3_s": float(q_volume[nearest]),
                "inventory_derivative_flux_L_s": float(q_volume[nearest] * 1000.0),
                "same_sign": bool(kh_q * q_volume[nearest] > 0.0),
            }
        )

    output = {
        "case": "Vasconcelos and Wright (2011), Case A",
        "source": str(VTK_ROOT.relative_to(CASE)).replace("\\", "/"),
        "report_window_s": [REPORT_START, REPORT_END],
        "sample_interval_s": float(np.median(np.diff(times))),
        "mapping_assumptions": {
            "planar_to_circular": (
                "Preserve tower section-averaged alpha.water, calculate an equivalent liquid "
                "height, then multiply by the physical circular tower area."
            ),
            "positive_flux_direction": "upward from horizontal pipe into riser",
            "volume_derivative_proxy": (
                "d/dt of mapped total tower liquid volume using a 0.25 s cubic "
                "Savitzky-Golay derivative. It equals base liquid flux while the open top is dry."
            ),
            "horizontal_depth": "D times section-averaged alpha.water in each x column",
            "horizontal_wave_decomposition": (
                "0.025 m resolved-profile pass, minus a 0.30 m spatial trend; envelope is 0.08 m RMS."
            ),
            "active_length": (
                "largest connected interval whose residual RMS envelope is at least 2 mm; "
                "also reported at half of each frame's maximum envelope."
            ),
        },
        "tower_geometry": {
            "diameter_m": DT,
            "physical_area_m2": AT,
            "height_m": TOWER_HEIGHT,
            "first_sample_row_height_above_crown_m": base_y - CROWN_Y,
        },
        "volume_derivative_flux": {
            "7.5_to_10.5_s": _window_summary(times, q_volume, 7.5, 10.5),
            "8.0_to_10.0_s": _window_summary(times, q_volume, 8.0, 10.0),
        },
        "temporal_smoothing_sensitivity": smoothing_sensitivity,
        "single_frame_velocity_cross_check": direct_cross_check,
        "open_top_dryness_check": {
            "maximum_top_row_water_fraction": float(
                max(float(row["top_row_water_fraction"]) for row, keep in zip(records, report) if keep)
            ),
            "mean_top_row_water_fraction": float(
                np.mean([float(row["top_row_water_fraction"]) for row, keep in zip(records, report) if keep])
            ),
            "note": (
                "No velocity was exported in VTK_CASEA_HTML. A negligible top-row water fraction "
                "supports treating dV_liquid/dt as the T-junction base liquid flux."
            ),
        },
        "riser_free_surface_lift": _level_lift(),
        "horizontal_wave": {
            "analysis_interval_m": [WAVE_X_MIN, WAVE_X_MAX],
            "snapshots": wave_snapshots,
            "7.5_to_10.5_s_residual_a90_median_m": float(np.median(residual_a90[report])),
            "7.5_to_10.5_s_residual_a90_max_m": float(np.max(residual_a90[report])),
            "7.5_to_10.5_s_active_length_median_m": float(np.median(active_length[report])),
            "7.5_to_10.5_s_active_length_max_m": float(np.max(active_length[report])),
        },
        "trace": [row for row, keep in zip(records, report) if keep],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(json.dumps({
        "volume_derivative_flux": output["volume_derivative_flux"],
        "temporal_smoothing_sensitivity": output["temporal_smoothing_sensitivity"],
        "single_frame_velocity_cross_check": output["single_frame_velocity_cross_check"],
        "open_top_dryness_check": output["open_top_dryness_check"],
        "riser_free_surface_lift": output["riser_free_surface_lift"],
        "horizontal_wave": output["horizontal_wave"],
    }, indent=2))


if __name__ == "__main__":
    main()
