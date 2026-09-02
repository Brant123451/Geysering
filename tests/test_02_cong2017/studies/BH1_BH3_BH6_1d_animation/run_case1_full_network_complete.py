#!/usr/bin/env python3
"""Run complete Campaign-2 events with the clean frozen Campaign-1 core.

The exact criterion-map source that generated the manuscript's frozen Series-B
classification is loaded by hash and executed without source transformation.
Its 2% near-dry regularisation film is retained for numerical stability but is
excluded from physical wetting-front detection and from later rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]
CASES_ROOT = TEST_ROOT / "cases"
MODEL_PATH = (
    TEST_ROOT
    / "studies"
    / "criterion_map"
    / "model"
    / "cong2017_network_twofluid.py"
)
EXPECTED_MODEL_SHA256 = (
    "cea1ffbf6dc5dbae38ab98205f08dbba4544a3e8f951d56de3eb944f3cd9ca23"
)
OUTPUT_ROOT = HERE / "case1_frozen_complete" / "model_1d"

PIPE_DIAMETER = 0.050
PAPER_H0_FROM_INVERT = 0.660
APPLIED_CONSTANT_HEAD = PAPER_H0_FROM_INVERT
TUNNEL_LENGTH = 6.590
RISER_X = 3.470
VALVE_X = 5.980
RISER_HEIGHT = 1.800
VALVE_OPEN_TIME = 0.250

CASES = {
    "BH1": {"paper_run": "B-H1", "Dr": 0.016, "experiment_geyser": True},
    "BH3": {"paper_run": "B-H3", "Dr": 0.026, "experiment_geyser": True},
    "BH6": {"paper_run": "B-H6", "Dr": 0.041, "experiment_geyser": False},
}


def load_network():
    name = "cong2017_case1_frozen_complete"
    source_bytes = MODEL_PATH.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"frozen model hash changed: expected {EXPECTED_MODEL_SHA256}, "
            f"found {source_hash}"
        )
    source = source_bytes.decode("utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(MODEL_PATH)
    sys.modules[name] = module
    exec(compile(source, str(MODEL_PATH), "exec"), module.__dict__)
    return module


def first_true(times: np.ndarray, condition: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return None if indices.size == 0 else float(times[int(indices[0])])


def final_window_diagnostics(
    times: np.ndarray,
    values: np.ndarray,
    *,
    duration: float = 2.0,
) -> dict[str, float]:
    mask = times >= times[-1] - duration
    t = times[mask]
    y = values[mask]
    slope = float(np.polyfit(t, y, 1)[0]) if t.size >= 2 else math.nan
    return {
        "duration_s": float(times[-1] - times[mask][0]),
        "minimum_m": float(np.nanmin(y)),
        "maximum_m": float(np.nanmax(y)),
        "range_m": float(np.nanmax(y) - np.nanmin(y)),
        "linear_slope_m_per_s": slope,
    }


def run_case(case_key: str, t_end: float) -> dict[str, object]:
    spec = CASES[case_key]
    module = load_network()
    case = module.NetworkCase(
        D=PIPE_DIAMETER,
        Dr=float(spec["Dr"]),
        riser_height=RISER_HEIGHT,
        L_up=RISER_X,
        L_mid=VALVE_X - RISER_X,
        L_down=TUNNEL_LENGTH - VALVE_X,
        x_riser_at=RISER_X,
        pocket_downstream=True,
        reservoir_head=APPLIED_CONSTANT_HEAD,
        air_head=0.0,
        init_water_level=APPLIED_CONSTANT_HEAD,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        valve_open_time=VALVE_OPEN_TIME,
        ds=0.020,
        dz=0.010,
        t_end=float(t_end),
    )

    output = OUTPUT_ROOT / case_key
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{case_key}] frozen Case-1 full-network {t_end:g} s run started", flush=True)
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
        "tun_gas_mass": np.asarray(record["tun_gas_mass"], dtype=float),
        "tun_gas_vol": np.asarray(record["tun_gas_vol"], dtype=float),
        "tot_liq": np.asarray(record["tot_liq"], dtype=float),
        "xt": np.asarray(record["xt"], dtype=float),
        "zr": np.asarray(record["zr"], dtype=float),
        "dx": np.asarray([record["dx"]], dtype=float),
        "dz": np.asarray([record["dz"]], dtype=float),
    }
    # The frozen core archives tunnel and riser liquid as area fractions.
    void_fraction = np.clip(1.0 - arrays["frames_alt"], 0.0, 1.0)
    release_mask = arrays["xt"] > VALVE_X
    wetting_front = []
    gas_nose = []
    for liquid, void in zip(arrays["frames_alt"], void_fraction):
        wet = np.flatnonzero(release_mask & (liquid > 0.0201))
        wetting_front.append(
            VALVE_X if wet.size == 0 else float(arrays["xt"][wet[-1]])
        )
        pocket = np.flatnonzero(void > 0.05)
        gas_nose.append(
            TUNNEL_LENGTH if pocket.size == 0 else float(arrays["xt"][pocket[0]])
        )
    arrays["frames_release_wetting_front_x"] = np.asarray(
        wetting_front, dtype=float
    )
    arrays["frames_gas_nose_x"] = np.asarray(gas_nose, dtype=float)
    np.savez_compressed(output / "case1_full_network_frames.npz", **arrays)

    t = arrays["t"]
    wtop = arrays["wtop"]
    tf = arrays["frames_t"]
    dx = float(arrays["dx"][0])
    rim = first_true(t, wtop >= RISER_HEIGHT - 0.5 * case.dz)
    arrival = first_true(tf, arrays["frames_gas_nose_x"] <= RISER_X + 0.5 * dx)
    wetting_complete = first_true(
        tf,
        arrays["frames_release_wetting_front_x"] >= TUNNEL_LENGTH - 0.5 * dx,
    )
    peak_index = int(np.nanargmax(wtop))
    final_window = final_window_diagnostics(t, wtop)
    event_has_tail = float(t[-1] - t[peak_index]) >= 3.0
    not_still_climbing = final_window["linear_slope_m_per_s"] <= 0.02
    physical_event_complete = bool(
        arrival is not None and event_has_tail and not_still_climbing
    )

    summary = {
        "case": case_key,
        "paper_run": spec["paper_run"],
        "variant": "frozen_case1_full_network_complete_v3",
        "status": "completed" if tf[-1] >= t_end - 1.0e-9 else "incomplete",
        "runtime_s": runtime,
        "source_network": str(MODEL_PATH),
        "model_provenance": {
            "frozen_source_sha256": EXPECTED_MODEL_SHA256,
            "horizontal_release": "frozen local FV fluxes with 2% near-dry numerical film",
            "subsequent_coupling": "Campaign-1 frozen fully synchronous pipe/tee/riser network",
            "external_case1_adapter": False,
            "per_case_parameter_fitting": False,
            "source_transformation": None,
            "physical_wet_threshold_area_fraction": 0.0201,
        },
        "paper_conditions": {
            "pipe_diameter_m": PIPE_DIAMETER,
            "riser_diameter_m": float(spec["Dr"]),
            "reservoir_head_from_invert_m": PAPER_H0_FROM_INVERT,
            "applied_constant_head_m": APPLIED_CONSTANT_HEAD,
            "pipe_length_m": TUNNEL_LENGTH,
            "riser_x_m": RISER_X,
            "valve_x_m": VALVE_X,
            "initial_air_reach_length_m": TUNNEL_LENGTH - VALVE_X,
            "initial_air_gauge_head_m": 0.0,
            "valve_opening_duration_s": VALVE_OPEN_TIME,
            "paper_manual_opening_duration_s_approx": 0.20,
        },
        "events": {
            "wetting_front_reaches_closed_cap_s": wetting_complete,
            "gas_nose_reaches_riser_s": arrival,
            "riser_surface_reaches_rim_s": rim,
            "maximum_riser_level_time_s": float(t[peak_index]),
        },
        "outcome": {
            "model_geyser": rim is not None,
            "experiment_geyser": bool(spec["experiment_geyser"]),
            "classification_match": bool(rim is not None)
            == bool(spec["experiment_geyser"]),
            "maximum_riser_level_m": float(wtop[peak_index]),
            "maximum_pocket_head_m": float(np.nanmax(arrays["pocket_head"])),
        },
        "completion": {
            "physical_event_complete": physical_event_complete,
            "minimum_post_peak_tail_s": 3.0,
            "post_peak_tail_s": float(t[-1] - t[peak_index]),
            "final_2s_riser": final_window,
        },
        "audit": {
            "liquid_floor_created_m3": float(record["dbg_created"]["t_floor"]),
            "frame_count": int(tf.size),
            "end_time_s": float(tf[-1]),
        },
        "artifacts": {
            "fields": "case1_full_network_frames.npz",
            "summary": "summary.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{case_key}] complete in {runtime:.1f} s; arrival={arrival}; "
        f"rim={rim}; max={wtop[peak_index]:.3f} m; "
        f"physical_end={physical_event_complete}",
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
