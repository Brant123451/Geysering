#!/usr/bin/env python3
"""Run B-H1/B-H3/B-H6 with the Campaign-1 horizontal production core.

The historical ``repaired`` results are deliberately left untouched.  This
variant writes only below ``case1_model_rerun`` and records the exact Case-1
source hashes used by the mirror adapter.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]
CASES_ROOT = TEST_ROOT / "cases"
MODEL_PATH = (
    CASES_ROOT
    / "BH1_Dr16_H066_L061"
    / "model"
    / "cong2017_network_twofluid.py"
)
OUTPUT_ROOT = HERE / "case1_release_full_event" / "model_1d"

PIPE_DIAMETER = 0.050
PAPER_H0_FROM_INVERT = 0.660
MODEL_H0_FROM_CROWN = PAPER_H0_FROM_INVERT - PIPE_DIAMETER
TUNNEL_LENGTH = 6.590
RISER_X = 3.470
VALVE_X = 5.980

CASES = {
    "BH1": {"paper_run": "B-H1", "Dr": 0.016, "experiment_geyser": True},
    "BH3": {"paper_run": "B-H3", "Dr": 0.026, "experiment_geyser": True},
    "BH6": {"paper_run": "B-H6", "Dr": 0.041, "experiment_geyser": False},
}


def load_campaign2_network():
    module_name = "cong2017_network_twofluid_case1_horizontal_variant"
    spec = importlib.util.spec_from_file_location(module_name, MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def first_true(times: np.ndarray, condition: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return None if indices.size == 0 else float(times[int(indices[0])])


def run_case(case_key: str, t_end: float) -> dict[str, object]:
    from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal

    spec = CASES[case_key]
    module = load_campaign2_network()
    ds = 0.020
    n_tunnel = max(20, int(round(TUNNEL_LENGTH / ds)))
    dx = TUNNEL_LENGTH / n_tunnel
    horizontal = Campaign2Case1MirroredHorizontal(
        length=TUNNEL_LENGTH,
        diameter=PIPE_DIAMETER,
        physical_valve_x=VALVE_X,
        physical_riser_x=RISER_X,
        initial_water_head_from_invert=PAPER_H0_FROM_INVERT,
        dx=dx,
        wave_speed=28.0,
        valve_open_time=0.20,
        coupling_interval=0.005,
    )
    case = module.NetworkCase(
        D=PIPE_DIAMETER,
        Dr=float(spec["Dr"]),
        riser_height=1.80,
        L_up=3.47,
        L_mid=2.51,
        L_down=0.61,
        x_riser_at=RISER_X,
        pocket_downstream=True,
        reservoir_head=MODEL_H0_FROM_CROWN,
        air_head=0.0,
        init_water_level=MODEL_H0_FROM_CROWN,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        valve_open_time=0.20,
        ds=ds,
        dz=0.010,
        t_end=float(t_end),
        case1_horizontal_solver=horizontal,
    )

    output = OUTPUT_ROOT / case_key
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[{case_key}] Case-1 release-wave/full-network {t_end:g} s run started",
        flush=True,
    )
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
        "frames_gas_nose_x": np.asarray(
            record["frames_gas_nose_x"], dtype=float
        ),
        "tun_gas_mass": np.asarray(record["tun_gas_mass"], dtype=float),
        "tun_gas_vol": np.asarray(record["tun_gas_vol"], dtype=float),
        "tot_liq": np.asarray(record["tot_liq"], dtype=float),
        "xt": np.asarray(record["xt"], dtype=float),
        "zr": np.asarray(record["zr"], dtype=float),
        "dx": np.asarray([record["dx"]], dtype=float),
        "dz": np.asarray([record["dz"]], dtype=float),
    }
    np.savez_compressed(output / "case1_horizontal_1d_frames.npz", **arrays)

    arrival = first_true(
        arrays["frames_t"],
        arrays["frames_gas_nose_x"] <= RISER_X + 0.5 * dx,
    )
    wetting_complete = first_true(
        arrays["frames_t"],
        arrays["frames_release_wetting_front_x"]
        >= TUNNEL_LENGTH - 0.5 * dx,
    )
    rim = first_true(
        arrays["t"],
        arrays["wtop"] >= case.riser_height - 0.5 * case.dz,
    )
    source_mass = float(
        horizontal.initial_state().gas.mass
    )
    final_mass = float(arrays["tun_gas_mass"][-1]) if arrays["tun_gas_mass"].size else math.nan
    summary = {
        "case": case_key,
        "paper_run": spec["paper_run"],
        "variant": "case1_release_wave_then_full_network_v4",
        "status": "completed" if arrays["frames_t"][-1] >= t_end - 1.0e-9 else "incomplete",
        "runtime_s": runtime,
        "source_network": str(MODEL_PATH),
        "horizontal_provenance": record["case1_horizontal_provenance"],
        "paper_conditions": {
            "pipe_diameter_m": PIPE_DIAMETER,
            "riser_diameter_m": float(spec["Dr"]),
            "reservoir_head_from_invert_m": PAPER_H0_FROM_INVERT,
            "model_head_from_pipe_crown_m": MODEL_H0_FROM_CROWN,
            "riser_x_m": RISER_X,
            "valve_x_m": VALVE_X,
            "initial_air_reach_length_m": TUNNEL_LENGTH - VALVE_X,
            "initial_air_gauge_head_m": 0.0,
            "valve_opening_duration_s": 0.20,
        },
        "events": {
            "wetting_front_reaches_closed_cap_s": wetting_complete,
            "case1_to_campaign2_handoff_s": record[
                "case1_horizontal_handoff_time"
            ],
            "gas_nose_reaches_riser_s": arrival,
            "riser_surface_reaches_rim_s": rim,
        },
        "outcome": {
            "model_geyser": rim is not None,
            "experiment_geyser": bool(spec["experiment_geyser"]),
            "maximum_riser_level_m": float(np.nanmax(arrays["wtop"])),
            "maximum_pocket_head_m": float(
                np.nanmax(arrays["pocket_head"])
            ),
        },
        "audit": {
            "initial_horizontal_gas_mass_kg": source_mass,
            "final_horizontal_gas_mass_kg": final_mass,
            "liquid_floor_created_m3": float(record["dbg_created"]["t_floor"]),
            "frame_count": int(arrays["frames_t"].size),
            "end_time_s": float(arrays["frames_t"][-1]),
        },
        "artifacts": {
            "fields": "case1_horizontal_1d_frames.npz",
            "summary": "summary.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{case_key}] complete in {runtime:.1f} s; "
        f"handoff={summary['events']['case1_to_campaign2_handoff_s']}; "
        f"geyser={summary['outcome']['model_geyser']}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), required=True)
    parser.add_argument("--t-end", type=float, default=20.0)
    args = parser.parse_args()
    if not math.isfinite(args.t_end) or args.t_end <= 0.0:
        parser.error("--t-end must be positive and finite")
    selected = tuple(CASES) if args.case == "all" else (args.case,)
    summaries = [run_case(case_key, args.t_end) for case_key in selected]
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
