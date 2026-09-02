#!/usr/bin/env python3
"""Recompute the Case-B 1-D paper series with source-consistent observers.

The numerical state is advanced by the case-local frozen solver.  This script
does not alter that state.  It replaces the old display diagnostics with two
observers that match the quantities read from the experiment video:

* ``Yfs`` is the upper alpha_l + alpha_g = 0.5 crossing of the occupied tower
  column connected to the tower base;
* ``Yint`` is the upper alpha_g = 0.02 crossing of the injected gas core
  connected to the tower base.

No time shift, curve fit, prescribed trajectory, or value assignment from the
experimental data is used.  Candidate outputs are written below
``outputs/paper_series_recomputed`` so the archived manuscript inputs are not
overwritten until the result has been audited.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = CASE_ROOT / "model" / "vw2011_network_twofluid.py"
EXPERIMENT_PATH = CASE_ROOT / "data" / "digitized" / "fig8_caseB_levels_runs_v2.csv"
OUTPUT_DIR = CASE_ROOT / "outputs" / "paper_series_recomputed"
OUTPUT_CSV = OUTPUT_DIR / "caseB_model_series_observed.csv"
OUTPUT_METRICS = OUTPUT_DIR / "caseB_model_series_observed_metrics.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "caseB_model_series_observed_manifest.json"

G = 9.81
L = 0.610
DT = 0.0127
T_END = 8.70
OCCUPANCY_THRESHOLD = 0.50
GAS_THRESHOLD = 0.02
MAX_BRIDGE_CELLS = 1


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("case_b_frozen_solver_recomputed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upper_crossing(z: np.ndarray, values: np.ndarray, threshold: float, last_active: int) -> float:
    """Linearly locate the upper threshold crossing after ``last_active``."""
    dz = float(np.median(np.diff(z))) if z.size > 1 else 0.0
    if last_active >= values.size - 1:
        return float(z[last_active] + 0.5 * dz)
    v0 = float(values[last_active])
    v1 = float(values[last_active + 1])
    if abs(v1 - v0) < 1.0e-12:
        return float(z[last_active] + 0.5 * dz)
    fraction = float(np.clip((threshold - v0) / (v1 - v0), 0.0, 1.0))
    return float(z[last_active] + fraction * (z[last_active + 1] - z[last_active]))


def _base_connected_top(
    z: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    require_base: bool,
    max_bridge_cells: int = MAX_BRIDGE_CELLS,
) -> float:
    """Return the top of the thresholded component connected to the tower base.

    One sub-threshold cell may be bridged to tolerate a single mixed numerical
    cell.  A component that starts farther from the base is rejected so an
    isolated droplet or bubble cannot redefine the experimental interface.
    """
    values = np.asarray(values, dtype=float)
    active = np.isfinite(values) & (values >= threshold)
    if not np.any(active):
        return 0.0

    start = 0
    if not active[0]:
        candidates = np.flatnonzero(active[: max_bridge_cells + 2])
        if candidates.size == 0 or require_base:
            return 0.0
        start = int(candidates[0])

    last_active = start
    gap = 0
    for index in range(start, active.size):
        if active[index]:
            last_active = index
            gap = 0
        else:
            gap += 1
            if gap > max_bridge_cells:
                break
    return _upper_crossing(z, values, threshold, last_active)


def _tower_observers(record: dict) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(record["zr"], dtype=float)
    alpha_l = np.asarray(record["frames_alr"], dtype=float)
    alpha_g = np.asarray(record["frames_agr"], dtype=float)
    if alpha_l.shape != alpha_g.shape:
        raise ValueError(f"Tower phase arrays differ: {alpha_l.shape} vs {alpha_g.shape}")

    yfs = np.zeros(alpha_l.shape[0], dtype=float)
    yint = np.zeros(alpha_l.shape[0], dtype=float)
    for index, (liquid, gas) in enumerate(zip(alpha_l, alpha_g)):
        occupancy = np.clip(liquid + gas, 0.0, 1.0)
        yfs[index] = _base_connected_top(
            z,
            occupancy,
            OCCUPANCY_THRESHOLD,
            require_base=True,
        )
        yint[index] = _base_connected_top(
            z,
            gas,
            GAS_THRESHOLD,
            require_base=False,
        )
        yint[index] = min(yint[index], yfs[index])
    return yfs, yint


def _first_crossing(time: np.ndarray, values: np.ndarray, threshold: float, after: float = 3.0) -> float | None:
    indices = np.flatnonzero((time >= after) & np.isfinite(values) & (values >= threshold))
    return float(time[indices[0]]) if indices.size else None


def _load_experiment() -> list[dict[str, str]]:
    with EXPERIMENT_PATH.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _track_errors(time: np.ndarray, values: np.ndarray, observable: str) -> dict[str, float | int | None]:
    points = [
        row
        for row in _load_experiment()
        if row["kind"] == observable and row["role"] == "rising_track"
    ]
    xp = np.asarray([float(row["Tstar"]) for row in points], dtype=float)
    yp = np.asarray([float(row["Ystar"]) for row in points], dtype=float)
    valid_window = np.isfinite(time) & np.isfinite(values)
    if np.count_nonzero(valid_window) < 2:
        return {"n": 0, "rmse": None, "bias": None}
    prediction = np.interp(xp, time[valid_window], values[valid_window], left=np.nan, right=np.nan)
    valid = np.isfinite(prediction)
    error = prediction[valid] - yp[valid]
    return {
        "n": int(np.count_nonzero(valid)),
        "rmse": float(np.sqrt(np.mean(error * error))) if error.size else None,
        "bias": float(np.mean(error)) if error.size else None,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_module(MODEL_PATH)
    case = model.NetworkCase(
        Dr=DT,
        air_head=0.610,
        init_water_level=0.356,
        t_end=T_END,
    )
    print(f"Running current case-local solver to t={T_END:.2f} s...", flush=True)
    record = model.run_network(case, verbose=True)

    time_s = np.asarray(record["frames_t"], dtype=float)
    yfs_m, yint_m = _tower_observers(record)
    tstar = time_s * math.sqrt(G * DT) / L
    yfs_star = yfs_m / L
    yint_star = yint_m / L

    pressure = np.concatenate([[np.nan], np.asarray(record["tr_head"], dtype=float)])
    pressure = pressure[: time_s.size] / L
    pocket = np.asarray(record["up_head"], dtype=float)[: time_s.size] / L
    old_yfs = np.asarray(record["wtop"], dtype=float)[: time_s.size] / L
    old_yint = np.asarray(record["itop"], dtype=float)[: time_s.size] / L

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t_s",
                "Tstar",
                "Yfs_star",
                "Yint_star",
                "Yfs_star_legacy_observer",
                "Yint_star_legacy_observer",
                "pocket_Hstar",
                "transducer_Hstar",
            ]
        )
        for row in zip(time_s, tstar, yfs_star, yint_star, old_yfs, old_yint, pocket, pressure):
            writer.writerow([f"{value:.9g}" for value in row])

    metrics = {
        "time_shift_applied": False,
        "curve_fit_applied": False,
        "observer": {
            "Yfs": f"base-connected alpha_l+alpha_g={OCCUPANCY_THRESHOLD:g} upper crossing",
            "Yint": f"base-connected injected-gas alpha_g={GAS_THRESHOLD:g} upper crossing",
            "max_bridge_cells": MAX_BRIDGE_CELLS,
        },
        "events_Tstar": {
            "Yint_0p02": _first_crossing(tstar, yint_star, 0.02),
            "Yint_0p10": _first_crossing(tstar, yint_star, 0.10),
            "Yint_0p50": _first_crossing(tstar, yint_star, 0.50),
            "Yint_0p80": _first_crossing(tstar, yint_star, 0.80),
            "Yfs_0p98": _first_crossing(tstar, yfs_star, 0.98),
            "Yfs_1p00": _first_crossing(tstar, yfs_star, 1.00),
        },
        "experiment_rising_track_error": {
            "Yfs": _track_errors(tstar, yfs_star, "fs"),
            "Yint": _track_errors(tstar, yint_star, "int"),
        },
        "legacy_observer_rising_track_error": {
            "Yfs": _track_errors(tstar, old_yfs, "fs"),
            "Yint": _track_errors(tstar, old_yint, "int"),
        },
        "pre_entry_Yfs_star_median_Tstar_3p0_to_3p35": float(
            np.median(yfs_star[(tstar >= 3.0) & (tstar <= 3.35)])
        ),
        "mass_diagnostics": {
            "initial_total_liquid_m3": float(record["tot_liq"][0]),
            "final_total_liquid_m3": float(record["tot_liq"][-1]),
            "relative_total_liquid_change": float(
                (record["tot_liq"][-1] - record["tot_liq"][0]) / record["tot_liq"][0]
            ),
        },
    }
    OUTPUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "case": "VW2011 Test B",
        "conditions": {"Dt_m": DT, "Ha0_m": 0.610, "Yfs0_m": 0.356},
        "solver": str(MODEL_PATH.relative_to(CASE_ROOT)),
        "solver_sha256": _sha256(MODEL_PATH),
        "script_sha256": _sha256(Path(__file__)),
        "experimental_input": str(EXPERIMENT_PATH.relative_to(CASE_ROOT)),
        "experimental_input_sha256": _sha256(EXPERIMENT_PATH),
        "outputs": [str(OUTPUT_CSV.relative_to(CASE_ROOT)), str(OUTPUT_METRICS.relative_to(CASE_ROOT))],
        "scientific_guards": [
            "native time retained",
            "no experimental values enter the numerical state or observers",
            "no curve fit or prescribed trajectory",
        ],
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
