#!/usr/bin/env python3
"""Audit one completed Cong (2017) Campaign-2 2D event.

The same definitions are intentionally used for every case.  The script is a
read-only consumer of archived metrics, stored centreline samples, and the
case-independent physical-rim report.  It does not inspect rendered HTML and
does not modify an OpenFOAM case.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RHO_WATER_KG_M3 = 998.0
GRAVITY_M_S2 = 9.81
ATMOSPHERE_PA = 101325.0
SUPPORTED_RELATIVE_ERROR = 0.10
PARTIAL_RELATIVE_ERROR = 0.30
CATCHUP_INITIAL_GAP_FRACTION = 0.02
CATCHUP_GRID_CELLS = 2.0
CATCHUP_CONSECUTIVE_SAMPLES = 3
PRESSURE_MIN_INITIAL_GAP_FRACTION = 0.10
PRESSURE_MIN_GRID_CELLS = 10.0
PRESSURE_SAMPLE_INTERVAL_S = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def relative_error(model: float | None, experiment: float | None) -> float | None:
    if model is None or experiment is None or experiment == 0.0:
        return None
    return abs(model - experiment) / abs(experiment)


def quantitative_support(model: float | None, experiment: float | None) -> str:
    error = relative_error(model, experiment)
    if error is None:
        return "missing"
    if error <= SUPPORTED_RELATIVE_ERROR + 1e-12:
        return "supported"
    if error <= PARTIAL_RELATIVE_ERROR + 1e-12:
        return "partial"
    return "failed"


def finite_interp(t: np.ndarray, y: np.ndarray, target: float) -> float | None:
    keep = np.isfinite(t) & np.isfinite(y)
    if not np.any(keep):
        return None
    return float(np.interp(target, t[keep], y[keep]))


def first_confirmed_catchup(
    t: np.ndarray,
    yfs: np.ndarray,
    yint: np.ndarray,
    ta_s: float,
    riser_dz_m: float,
) -> tuple[float | None, float | None, float | None]:
    gap = yfs - yint
    start = np.flatnonzero((t >= ta_s) & np.isfinite(gap))
    if start.size == 0:
        return None, None, None
    initial_gap = float(gap[int(start[0])])
    threshold = max(
        CATCHUP_INITIAL_GAP_FRACTION * initial_gap,
        CATCHUP_GRID_CELLS * riser_dz_m,
    )
    candidate = (t >= ta_s) & np.isfinite(gap) & (gap <= threshold)
    for index in range(candidate.size - CATCHUP_CONSECUTIVE_SAMPLES + 1):
        if np.all(candidate[index : index + CATCHUP_CONSECUTIVE_SAMPLES]):
            return float(t[index]), threshold, initial_gap
    return None, threshold, initial_gap


def crossings(z: np.ndarray, alpha: np.ndarray) -> tuple[list[float], list[float]]:
    water_to_air: list[float] = []
    air_to_water: list[float] = []
    for index in range(alpha.size - 1):
        a0 = float(alpha[index])
        a1 = float(alpha[index + 1])
        if (a0 - 0.5) * (a1 - 0.5) > 0.0 or abs(a1 - a0) < 1e-12:
            continue
        fraction = (0.5 - a0) / (a1 - a0)
        location = float(z[index] + fraction * (z[index + 1] - z[index]))
        if a0 >= 0.5 > a1:
            water_to_air.append(location)
        elif a0 <= 0.5 < a1:
            air_to_water.append(location)
    return water_to_air, air_to_water


def centreline_bubble_state(path: Path) -> dict[str, float] | None:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 3:
        return None
    order = np.argsort(data[:, 0])
    data = data[order]
    keep = np.r_[True, np.diff(data[:, 0]) > 1e-12]
    data = data[keep]
    # The OpenFOAM sampled line begins 1 mm above the pipe crown.
    z_above_crown = 0.001 + data[:, 0]
    alpha = np.clip(data[:, 1], 0.0, 1.0)
    pressure_pa = data[:, 2]
    water_to_air, air_to_water = crossings(z_above_crown, alpha)
    if not water_to_air or not air_to_water:
        return None
    yfs = max(water_to_air)
    yint = max(air_to_water)
    gap = yfs - yint
    if gap <= 0.0:
        return None

    gas_indices = np.flatnonzero((z_above_crown < yint) & (alpha < 0.5))
    if gas_indices.size == 0:
        return None
    high = int(gas_indices[-1])
    low = high
    while low > 0 and alpha[low - 1] < 0.5:
        low -= 1
    while high + 1 < alpha.size and alpha[high + 1] < 0.5:
        high += 1
    gas_weight = np.clip(1.0 - alpha[low : high + 1], 0.0, 1.0)
    if not np.any(gas_weight > 0.0):
        return None
    pocket_pressure_pa = float(
        np.average(pressure_pa[low : high + 1], weights=gas_weight)
    )
    pocket_head_m = (pocket_pressure_pa - ATMOSPHERE_PA) / (
        RHO_WATER_KG_M3 * GRAVITY_M_S2
    )
    return {
        "Yfs_m_above_crown": yfs,
        "Yint_m_above_crown": yint,
        "water_column_length_m": gap,
        "pocket_pressure_Pa_abs": pocket_pressure_pa,
        "pocket_head_m_water": pocket_head_m,
        "pocket_head_over_water_column": pocket_head_m / gap,
    }


def select_time_directories(
    root: Path, start_s: float, end_s: float
) -> list[tuple[float, Path]]:
    available: list[tuple[float, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            time_s = float(child.name)
        except ValueError:
            continue
        if start_s <= time_s <= end_s:
            available.append((time_s, child))
    available.sort(key=lambda item: item[0])
    selected: list[tuple[float, Path]] = []
    next_time = start_s
    for time_s, child in available:
        if time_s + 1e-9 < next_time:
            continue
        selected.append((time_s, child))
        next_time = time_s + PRESSURE_SAMPLE_INTERVAL_S
    return selected


def percentile_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def sample_bubble_pressure(
    root: Path,
    start_s: float,
    end_s: float,
    initial_gap_m: float,
    riser_dz_m: float,
    output_csv: Path,
) -> dict[str, Any]:
    minimum_gap = max(
        PRESSURE_MIN_INITIAL_GAP_FRACTION * initial_gap_m,
        PRESSURE_MIN_GRID_CELLS * riser_dz_m,
    )
    rows: list[dict[str, float]] = []
    missing_files = 0
    for time_s, directory in select_time_directories(root, start_s, end_s):
        candidates = sorted(directory.glob("*alpha.water_p_U*"))
        if not candidates:
            missing_files += 1
            continue
        state = centreline_bubble_state(candidates[0])
        if state is None or state["water_column_length_m"] < minimum_gap:
            continue
        state = {"t_s": time_s, **state}
        rows.append(state)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_s",
        "Yfs_m_above_crown",
        "Yint_m_above_crown",
        "water_column_length_m",
        "pocket_pressure_Pa_abs",
        "pocket_head_m_water",
        "pocket_head_over_water_column",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "definition": (
            "gas-fraction-weighted pressure of the uppermost enclosed centreline "
            "gas component divided by the overlying water-column length"
        ),
        "sample_interval_s": PRESSURE_SAMPLE_INTERVAL_S,
        "minimum_resolved_water_column_m": minimum_gap,
        "sample_count": len(rows),
        "missing_sample_files": missing_files,
        "head_over_water_column": percentile_stats(
            [row["pocket_head_over_water_column"] for row in rows]
        ),
        "head_m_water": percentile_stats(
            [row["pocket_head_m_water"] for row in rows]
        ),
        "output_csv": str(output_csv),
    }


def pressure_probe_metrics(
    series: dict[str, np.ndarray], ta_s: float, event_end_s: float, h0_m: float
) -> dict[str, Any]:
    t = series["t_s"]
    ratio = series["head_m_water"] / h0_m

    def extrema(mask: np.ndarray) -> dict[str, float] | None:
        indices = np.flatnonzero(mask & np.isfinite(ratio))
        if indices.size == 0:
            return None
        low = int(indices[np.argmin(ratio[indices])])
        high = int(indices[np.argmax(ratio[indices])])
        return {
            "minimum_H_over_H0": float(ratio[low]),
            "minimum_time_s": float(t[low]),
            "maximum_H_over_H0": float(ratio[high]),
            "maximum_time_s": float(t[high]),
        }

    return {
        "definition": "PT1 gauge pressure head divided by H0",
        "global": extrema(np.ones_like(t, dtype=bool)),
        "arrival_to_event_end": extrema((t >= ta_s) & (t <= event_end_s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", type=Path, required=True)
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--rim-report", type=Path, required=True)
    parser.add_argument("--riser-series", type=Path, required=True)
    parser.add_argument("--pt1-series", type=Path, required=True)
    parser.add_argument("--centreline-root", type=Path, required=True)
    parser.add_argument("--riser-dz-m", type=float, required=True)
    parser.add_argument("--experiment-vnet-m-s", type=float, required=True)
    parser.add_argument("--experiment-vtaylor-m-s", type=float, required=True)
    parser.add_argument("--experiment-rim-time-s", type=float)
    parser.add_argument("--paper-break-earliest-s", type=float)
    parser.add_argument("--paper-break-latest-s", type=float)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-pressure-csv", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.case_config.read_text(encoding="utf-8"))
    archived_metrics = json.loads(args.model_metrics.read_text(encoding="utf-8"))
    rim = json.loads(args.rim_report.read_text(encoding="utf-8"))
    riser = load_csv(args.riser_series)
    pt1 = load_csv(args.pt1_series)

    experiment = config["experiment"]
    ta_model = archived_metrics["model"]["Ta_s"]
    if ta_model is None:
        raise ValueError("model Ta is missing")
    ta_model = float(ta_model)
    catchup_time, catchup_threshold, initial_gap = first_confirmed_catchup(
        riser["t_s"],
        riser["Yfs_m_above_crown"],
        riser["Yint_m_above_crown"],
        ta_model,
        args.riser_dz_m,
    )
    if initial_gap is None:
        raise ValueError("no post-arrival resolved water-column gap")

    rim_time = rim["decision"]["first_full_gate_time_s"]
    candidates = [float(riser["t_s"][-1])]
    if rim_time is not None:
        candidates.append(float(rim_time))
    if catchup_time is not None:
        candidates.append(float(catchup_time))
    event_end = min(candidates)

    yfs_start = finite_interp(
        riser["t_s"], riser["Yfs_m_above_crown"], ta_model
    )
    yint_start = finite_interp(
        riser["t_s"], riser["Yint_m_above_crown"], ta_model
    )
    yfs_end = finite_interp(
        riser["t_s"], riser["Yfs_m_above_crown"], event_end
    )
    yint_end = finite_interp(
        riser["t_s"], riser["Yint_m_above_crown"], event_end
    )
    if rim_time is not None and math.isclose(event_end, float(rim_time), abs_tol=1e-9):
        yfs_end = float(config["physical_geometry_m"]["riser_height_above_pipe_crown"])

    duration = event_end - ta_model
    if duration <= 0.0:
        raise ValueError("non-positive event duration")
    vfs_model = None if yfs_start is None or yfs_end is None else (yfs_end - yfs_start) / duration
    vint_model = None if yint_start is None or yint_end is None else (yint_end - yint_start) / duration
    vnet_model = None if vfs_model is None or vint_model is None else vint_model - vfs_model

    model_classification = rim["decision"]["classification"]
    experiment_geyser = str(experiment["classification"]).upper() == "GEYSER"
    model_geyser = model_classification == "GEYSER"
    h0_m = float(config["initial_conditions"]["H0_m_above_pipe_invert"])
    bubble_pressure = sample_bubble_pressure(
        args.centreline_root,
        ta_model,
        event_end,
        initial_gap,
        args.riser_dz_m,
        args.output_pressure_csv,
    )
    pressure_probe = pressure_probe_metrics(pt1, ta_model, event_end, h0_m)

    yfs = riser["Yfs_m_above_crown"]
    yint = riser["Yint_m_above_crown"]
    valid_after_arrival = (riser["t_s"] >= ta_model) & np.isfinite(yfs)
    peak_index = int(np.flatnonzero(valid_after_arrival)[np.argmax(yfs[valid_after_arrival])])
    valid_yint_event = (
        (riser["t_s"] >= ta_model)
        & (riser["t_s"] <= event_end)
        & np.isfinite(yint)
    )
    peak_yint_index = int(
        np.flatnonzero(valid_yint_event)[np.argmax(yint[valid_yint_event])]
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "case_id": config["case_id"],
        "paper_run": config["paper_run"],
        "audit_scope": "completed 2D event versus Cong et al. (2017)",
        "uniform_rules": {
            "time_origin": "seconds after valve opening begins",
            "quantitative_support": {
                "supported": "absolute relative error <= 10%",
                "partial": "10% < absolute relative error <= 30%",
                "failed": "absolute relative error > 30%",
                "missing": "no genuinely comparable value",
            },
            "arrival": "first resolved gas nose 5 mm above pipe crown",
            "classification": "case-independent physical-rim report, not 98% level or image",
            "catchup_proxy": (
                "first three consecutive trajectory samples with Yfs-Yint <= "
                "max(2% of initial post-arrival gap, 2 riser dz)"
            ),
            "event_end": "earliest of physical-rim crossing, catch-up proxy, stored event end",
            "average_interface_speed": "endpoint displacement divided by event duration",
            "net_speed": "vint-vfs, equivalent to d[Yfs0-(Yfs-Yint)]/dt",
            "bubble_pressure": bubble_pressure["definition"],
        },
        "source_types": {
            "experiment": "published Table 2 / figures / prose",
            "model_trajectory": "2D-derived centreline alpha=0.5 crossings",
            "model_pressure": "2D-derived centreline gas pressure and archived PT1 probe",
            "classification": "stored-field physical-rim surface integration",
        },
        "paper_contract": {
            "D_m": config["physical_geometry_m"]["pipe_inner_diameter"],
            "Dr_m": config["physical_geometry_m"]["riser_inner_diameter"],
            "H0_m": h0_m,
            "L0_m": config["initial_conditions"]["pocket_length_m"],
            "valve_opening_s": 0.2,
            "planar_area_equivalent_width_m": config["planar_mapping"]["area_equivalent_riser_width_m"],
            "limitation": config["planar_mapping"]["limitation"],
        },
        "completion": {
            "normal_end": rim["coverage"]["solver_log"]["normal_end"],
            "fatal_error": rim["coverage"]["solver_log"]["fatal_error"],
            "last_computed_time_s": rim["coverage"]["last_computed_time_s"],
            "declared_observation_end_reached": rim["decision"]["declared_observation_end_reached"],
            "classification_final": rim["decision"]["final"],
        },
        "classification": {
            "experiment": "GEYSER" if experiment_geyser else "NO_GEYSER",
            "model": model_classification,
            "support": "supported" if experiment_geyser == model_geyser else "failed",
            "physical_rim_first_full_gate_time_s": rim_time,
            "maximum_rim_plane_alpha": rim["metrics"]["maximum_rim_plane_alpha"],
            "cumulative_positive_liquid_volume_m3": rim["metrics"]["cumulative_positive_liquid_volume_m3"],
        },
        "arrival": {
            "experiment_s": experiment["Ta_s"],
            "model_s": ta_model,
            "relative_error": relative_error(ta_model, float(experiment["Ta_s"])),
            "support": quantitative_support(ta_model, float(experiment["Ta_s"])),
        },
        "event": {
            "model_event_end_s": event_end,
            "model_event_duration_after_arrival_s": duration,
            "model_catchup_proxy_s": catchup_time,
            "catchup_gap_threshold_m": catchup_threshold,
            "initial_postarrival_gap_m": initial_gap,
            "paper_break_bracket_s": [
                args.paper_break_earliest_s,
                args.paper_break_latest_s,
            ],
            "experiment_rim_time_s": args.experiment_rim_time_s,
            "rim_time_relative_error": relative_error(rim_time, args.experiment_rim_time_s),
            "rim_time_support": quantitative_support(rim_time, args.experiment_rim_time_s),
            "model_peak_Yfs_m": float(yfs[peak_index]),
            "model_peak_Yfs_time_s": float(riser["t_s"][peak_index]),
            "model_peak_Yint_before_event_end_m": float(yint[peak_yint_index]),
            "model_peak_Yint_time_s": float(riser["t_s"][peak_yint_index]),
        },
        "interface_speeds": {
            "model_definition": "common event-endpoint averages; not claimed identical to the paper's undisclosed averaging implementation",
            "vfs": {
                "experiment_m_s": experiment["vfs_m_s"],
                "model_m_s": vfs_model,
                "relative_error": relative_error(vfs_model, float(experiment["vfs_m_s"])),
                "support": quantitative_support(vfs_model, float(experiment["vfs_m_s"])),
            },
            "vint": {
                "experiment_m_s": experiment["vint_m_s"],
                "model_m_s": vint_model,
                "relative_error": relative_error(vint_model, float(experiment["vint_m_s"])),
                "support": quantitative_support(vint_model, float(experiment["vint_m_s"])),
            },
            "vnet": {
                "experiment_m_s": args.experiment_vnet_m_s,
                "model_m_s": vnet_model,
                "relative_error": relative_error(vnet_model, args.experiment_vnet_m_s),
                "support": quantitative_support(vnet_model, args.experiment_vnet_m_s),
            },
            "vTaylor_experiment_m_s": args.experiment_vtaylor_m_s,
            "model_vnet_over_vTaylor": None if vnet_model is None else vnet_model / args.experiment_vtaylor_m_s,
        },
        "bubble_pressure": bubble_pressure,
        "pt1_pressure": pressure_probe,
        "input_provenance": {
            "case_config": {"path": str(args.case_config), "sha256": sha256(args.case_config)},
            "model_metrics": {"path": str(args.model_metrics), "sha256": sha256(args.model_metrics)},
            "rim_report": {"path": str(args.rim_report), "sha256": sha256(args.rim_report)},
            "riser_series": {"path": str(args.riser_series), "sha256": sha256(args.riser_series)},
            "pt1_series": {"path": str(args.pt1_series), "sha256": sha256(args.pt1_series)},
            "centreline_root": str(args.centreline_root),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
