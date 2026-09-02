#!/usr/bin/env python3
"""Recompute the selected Cong (2017) Series-B cases with the repaired dry reach.

All three cases use the same paper geometry, initial/boundary conditions and model
core.  Only the riser diameter changes.  Results are archived under this study so
the historical criterion-map outputs and the OpenFOAM cases are not overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]
CASES_ROOT = TEST_ROOT / "cases"
OUTPUT_ROOT = HERE / "repaired" / "model_1d"
PAPER_H0_INVERT = 0.66
PIPE_DIAMETER = 0.050
MODEL_H0_CROWN = PAPER_H0_INVERT - PIPE_DIAMETER

CASES = {
    "BH1": {
        "folder": "BH1_Dr16_H066_L061",
        "paper_run": "B-H1",
        "Dr": 0.016,
        "experiment_geyser": True,
    },
    "BH3": {
        "folder": "BH3_Dr26_H066_L061",
        "paper_run": "B-H3",
        "Dr": 0.026,
        "experiment_geyser": True,
    },
    "BH6": {
        "folder": "BH6_Dr41_H066_L061",
        "paper_run": "B-H6",
        "Dr": 0.041,
        "experiment_geyser": False,
    },
}


def load_model(case_key: str):
    folder = str(CASES[case_key]["folder"])
    model_path = CASES_ROOT / folder / "model" / "cong2017_network_twofluid.py"
    module_name = f"cong2017_network_twofluid_repaired_{case_key.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, model_path


def first_crossing(time_values: np.ndarray, values: np.ndarray, threshold: float):
    indices = np.flatnonzero(values >= threshold)
    return None if indices.size == 0 else float(time_values[indices[0]])


def normalized_source_hash(path: Path) -> str:
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def wetting_fronts(
    times: np.ndarray,
    liquid: np.ndarray,
    x: np.ndarray,
    dx: float,
    valve_x: float,
) -> tuple[np.ndarray, float | None]:
    release = x > valve_x
    release_x = x[release]
    fronts = np.full(times.size, valve_x, dtype=float)
    complete = None
    for index, profile in enumerate(liquid[:, release]):
        wet = np.flatnonzero(profile > 1.0e-12)
        if wet.size:
            fronts[index] = min(float(release_x[wet[-1]] + 0.5 * dx), float(x[-1] + 0.5 * dx))
        if complete is None and wet.size == release_x.size:
            complete = float(times[index])
    return fronts, complete


def run_case(case_key: str) -> dict[str, object]:
    spec = CASES[case_key]
    module, model_path = load_model(case_key)
    output = OUTPUT_ROOT / case_key
    output.mkdir(parents=True, exist_ok=True)

    case = module.NetworkCase(
        D=PIPE_DIAMETER,
        Dr=float(spec["Dr"]),
        riser_height=1.80,
        L_up=3.47,
        L_mid=2.51,
        L_down=0.61,
        x_riser_at=3.47,
        pocket_downstream=True,
        reservoir_head=MODEL_H0_CROWN,
        air_head=0.0,
        init_water_level=MODEL_H0_CROWN,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        valve_open_time=0.20,
        t_end=13.0,
    )

    print(f"[{case_key}] start repaired 13 s run", flush=True)
    started = time.perf_counter()
    record = module.run_network(case, verbose=True)
    runtime = time.perf_counter() - started

    arrays = {
        "t": np.asarray(record["t"], dtype=float),
        "wtop": np.asarray(record["wtop"], dtype=float),
        "itop": np.asarray(record["itop"], dtype=float),
        "pocket_head": np.asarray(record["pocket_head"], dtype=float),
        "up_head": np.asarray(record["up_head"], dtype=float),
        "frames_t": np.asarray(record["frames_t"], dtype=float),
        "frames_alt": np.asarray(record["frames_alt"], dtype=np.float32),
        "frames_alr": np.asarray(record["frames_alr"], dtype=np.float32),
        "frames_agr": np.asarray(record["frames_agr"], dtype=np.float32),
        "frames_itop": np.asarray(record["frames_itop"], dtype=float),
        "frames_core_mass": np.asarray(record["frames_core_mass"], dtype=float),
        "frames_release_wetting_front_x": np.asarray(
            record["frames_release_wetting_front_x"], dtype=float
        ),
        "frames_reflected_front_x": np.asarray(
            record["frames_reflected_front_x"], dtype=float
        ),
        "frames_gas_nose_x": np.asarray(record["frames_gas_nose_x"], dtype=float),
        "tun_gas_mass": np.asarray(record["tun_gas_mass"], dtype=float),
        "tun_gas_vol": np.asarray(record["tun_gas_vol"], dtype=float),
        "tot_liq": np.asarray(record["tot_liq"], dtype=float),
        "xt": np.asarray(record["xt"], dtype=float),
        "zr": np.asarray(record["zr"], dtype=float),
        "dx": np.asarray([record["dx"]], dtype=float),
        "dz": np.asarray([record["dz"]], dtype=float),
    }
    fronts, wetting_complete = wetting_fronts(
        arrays["frames_t"],
        arrays["frames_alt"],
        arrays["xt"],
        float(arrays["dx"][0]),
        case.x_valve,
    )
    arrays["wetting_front_x"] = fronts
    np.savez_compressed(output / "repaired_1d_frames.npz", **arrays)

    initial_release = arrays["frames_alt"][0, arrays["xt"] > case.x_valve]
    gas_mass = arrays["tun_gas_mass"]
    gas_mass_full_change = (
        float(gas_mass[-1] / gas_mass[0] - 1.0) if gas_mass.size > 1 else None
    )
    t_arrival = first_crossing(arrays["t"], arrays["itop"], 0.02)
    t_rim = first_crossing(arrays["t"], arrays["wtop"], 0.98 * case.riser_height)
    mass_times = arrays["t"][1 : 1 + gas_mass.size]
    wetting_mass_index = (
        int(np.argmin(np.abs(mass_times - wetting_complete)))
        if wetting_complete is not None and gas_mass.size
        else None
    )
    wetting_mass_drift = (
        float(gas_mass[wetting_mass_index] / gas_mass[0] - 1.0)
        if wetting_mass_index is not None
        else None
    )
    summary = {
        "schema_version": 3,
        "case": case_key,
        "paper_run": spec["paper_run"],
        "status": "COMPLETE" if float(arrays["t"][-1]) >= 12.999 else "PARTIAL",
        "runtime_s": round(runtime, 2),
        "source_model": str(model_path.relative_to(TEST_ROOT)).replace("\\", "/"),
        "source_model_sha256": normalized_source_hash(model_path),
        "geometry_m": {
            "D": case.D,
            "Dr": case.Dr,
            "riser_height": case.riser_height,
            "tank_to_riser": case.L_up,
            "riser_to_valve": case.L_mid,
            "valve_to_cap_L0": case.L_down,
            "total_pipe": case.L_tunnel,
        },
        "initial_and_boundary_conditions": {
            "paper_H0_above_pipe_invert_m": PAPER_H0_INVERT,
            "model_head_above_pipe_crown_m": MODEL_H0_CROWN,
            "initial_riser_level_above_pipe_crown_m": MODEL_H0_CROWN,
            "initial_release_reach_liquid_fraction_min": float(np.min(initial_release)),
            "initial_release_reach_liquid_fraction_max": float(np.max(initial_release)),
            "initial_pocket_gauge_head_m": case.air_head,
            "valve_opening_duration_s": case.valve_open_time,
        },
        "dimensionless_air_volume": case.V_air
        / (0.25 * math.pi * case.Dr**2 * PAPER_H0_INVERT),
        "wet_dry_checks": {
            "finite_wetting_front": True,
            "wetting_front_reaches_cap_s": wetting_complete,
            "model_release_cap_wetted_s": record["release_cap_wetted_time"],
            "reflected_front_moves_left": bool(
                np.any(
                    np.isfinite(arrays["frames_reflected_front_x"])
                    & (arrays["frames_reflected_front_x"] < case.L_tunnel - 0.5 * case.ds)
                )
            ),
            "minimum_reflected_front_x_m": (
                float(np.nanmin(arrays["frames_reflected_front_x"]))
                if np.any(np.isfinite(arrays["frames_reflected_front_x"]))
                else None
            ),
            "tunnel_gas_mass_relative_drift_through_wetting": wetting_mass_drift,
            "tunnel_gas_mass_relative_change_full_run_including_riser_venting": gas_mass_full_change,
            "liquid_floor_created_m3_full_run": float(record["dbg_created"]["t_floor"]),
        },
        "model_outcome": {
            "geyser": t_rim is not None,
            "experiment_geyser": bool(spec["experiment_geyser"]),
            "gas_arrival_at_riser_s": t_arrival,
            "free_surface_at_rim_s": t_rim,
            "maximum_riser_level_m": float(np.nanmax(arrays["wtop"])),
            "maximum_pocket_head_m": float(np.nanmax(arrays["pocket_head"])),
        },
        "artifacts": {"frames_npz": "repaired_1d_frames.npz"},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[{case_key}] complete in {runtime:.1f} s; "
        f"geyser={summary['model_outcome']['geyser']}; "
        f"wetting_complete={wetting_complete}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    args = parser.parse_args()
    summary = run_case(args.case)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
