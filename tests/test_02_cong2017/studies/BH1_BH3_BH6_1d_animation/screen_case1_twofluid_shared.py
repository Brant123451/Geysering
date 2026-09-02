#!/usr/bin/env python3
"""Screen one shared 1D closure with the Case-1 horizontal adapter.

This is a diagnostic runner, not a result generator.  It keeps the published
Campaign-2 geometry and initial/boundary conditions, injects the hash-locked
Case-1 horizontal core, and varies only *shared* vertical two-fluid closure
coefficients supplied explicitly on the command line.  Nothing in this file
prescribes a case outcome or a geyser height.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal
from run_case1_model_cases import load_campaign2_network


PIPE_DIAMETER = 0.050
PAPER_H0_FROM_INVERT = 0.660
MODEL_H0_FROM_CROWN = PAPER_H0_FROM_INVERT - PIPE_DIAMETER
TUNNEL_LENGTH = 6.590
RISER_X = 3.470
VALVE_X = 5.980
RISER_HEIGHT = 1.800
VALVE_OPEN_TIME = 0.200
TOP_LIQUID_OUTFLOW_TOLERANCE_M3 = 1.0e-9

CASES = {
    "BH1": {"Dr": 0.016, "experiment_geyser": True},
    "BH3": {"Dr": 0.026, "experiment_geyser": True},
    "BH6": {"Dr": 0.041, "experiment_geyser": False},
}


def run(args: argparse.Namespace) -> dict[str, object]:
    spec = CASES[args.case]
    module = load_campaign2_network()
    n_tunnel = max(20, int(round(TUNNEL_LENGTH / args.ds)))
    dx = TUNNEL_LENGTH / n_tunnel
    horizontal = Campaign2Case1MirroredHorizontal(
        length=TUNNEL_LENGTH,
        diameter=PIPE_DIAMETER,
        physical_valve_x=VALVE_X,
        physical_riser_x=RISER_X,
        initial_water_head_from_invert=PAPER_H0_FROM_INVERT,
        dx=dx,
        wave_speed=args.wave_speed,
        valve_open_time=VALVE_OPEN_TIME,
        gas_temperature=296.15,
        coupling_interval=args.coupling_interval,
    )
    case = module.NetworkCase(
        D=PIPE_DIAMETER,
        Dr=float(spec["Dr"]),
        riser_height=RISER_HEIGHT,
        L_up=RISER_X,
        L_mid=VALVE_X - RISER_X,
        L_down=TUNNEL_LENGTH - VALVE_X,
        x_riser_at=RISER_X,
        pocket_downstream=True,
        reservoir_head=MODEL_H0_FROM_CROWN,
        air_head=0.0,
        init_water_level=MODEL_H0_FROM_CROWN,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        valve_open_time=VALVE_OPEN_TIME,
        ds=args.ds,
        dz=args.dz,
        t_end=args.t_end,
        gas_drive_eff=args.gas_drive_eff,
        entry_drive_eff=args.entry_drive_eff,
        gas_escape_eff=args.gas_escape_eff,
        case1_horizontal_solver=horizontal,
        case1_handoff_event="riser_arrival",
    )
    started = time.perf_counter()
    record = module.run_network(case, verbose=args.verbose)
    runtime = time.perf_counter() - started
    times = np.asarray(record["t"], dtype=float)
    levels = np.asarray(record["wtop"], dtype=float)
    top_liquid_outflow = np.asarray(
        record["top_liquid_outflow"], dtype=float
    )
    top_boundary_outflow = np.asarray(
        record["top_liquid_boundary_outflow"], dtype=float
    )
    top_capacity_outflow = np.asarray(
        record["top_liquid_capacity_outflow"], dtype=float
    )
    rim_level = RISER_HEIGHT - 0.5 * float(record["dz"])
    passages = np.flatnonzero(
        top_liquid_outflow > TOP_LIQUID_OUTFLOW_TOLERANCE_M3
    )
    maximum_level_index = int(np.nanargmax(levels))
    provenance = record.get("case1_horizontal_provenance")
    return {
        "schema_version": 1,
        "role": "shared-closure diagnostic only",
        "case": args.case,
        "geometry": {
            "D_m": PIPE_DIAMETER,
            "Dr_m": float(spec["Dr"]),
            "riser_height_m": RISER_HEIGHT,
            "riser_x_m": RISER_X,
            "valve_x_m": VALVE_X,
            "cap_x_m": TUNNEL_LENGTH,
            "H0_from_invert_m": PAPER_H0_FROM_INVERT,
            "H0_from_crown_m": MODEL_H0_FROM_CROWN,
            "valve_open_time_s": VALVE_OPEN_TIME,
            "gas_temperature_K": 296.15,
        },
        "shared_closure": {
            "gas_drive_eff": args.gas_drive_eff,
            "entry_drive_eff": args.entry_drive_eff,
            "gas_escape_eff": args.gas_escape_eff,
            "wave_speed_m_s": args.wave_speed,
            "ds_m": args.ds,
            "dz_m": args.dz,
        },
        "result": {
            "end_time_s": float(times[-1]),
            "maximum_riser_level_m": float(np.nanmax(levels)),
            "maximum_riser_level_time_s": float(
                times[maximum_level_index]
            ),
            "final_riser_level_m": float(levels[-1]),
            "riser_top_cell_centre_m": rim_level,
            "cumulative_top_liquid_outflow_m3": float(
                top_liquid_outflow[-1]
            ),
            "cumulative_top_boundary_flux_m3": float(
                top_boundary_outflow[-1]
            ),
            "cumulative_top_capacity_overflow_m3": float(
                top_capacity_outflow[-1]
            ),
            "outflow_classification_tolerance_m3": (
                TOP_LIQUID_OUTFLOW_TOLERANCE_M3
            ),
            "geyser": bool(passages.size),
            "first_liquid_ejection_time_s": (
                None if passages.size == 0 else float(times[passages[0]])
            ),
            "experiment_geyser": bool(spec["experiment_geyser"]),
            "classification_match": bool(passages.size)
            == bool(spec["experiment_geyser"]),
            "runtime_s": runtime,
            "liquid_floor_created_m3": float(record["dbg_created"]["t_floor"]),
        },
        "model_provenance": {
            "case1_horizontal_used": bool(record.get("case1_horizontal_used")),
            "case1_handoff_time_s": record.get("case1_horizontal_handoff_time"),
            "case1_handoff_event": "riser_arrival",
            "case1_horizontal": provenance,
            "vertical": "resolved 1D two-fluid area/mass/momentum model",
            "per_case_parameter_fitting": None,
            "single_case_screen_cannot_prove_shared_parameters": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--gas-drive-eff", type=float, default=1.0)
    parser.add_argument("--entry-drive-eff", type=float, default=1.0)
    parser.add_argument("--gas-escape-eff", type=float, default=1.0)
    parser.add_argument("--wave-speed", type=float, default=28.0)
    parser.add_argument("--coupling-interval", type=float, default=0.005)
    parser.add_argument("--ds", type=float, default=0.040)
    parser.add_argument("--dz", type=float, default=0.020)
    parser.add_argument("--t-end", type=float, default=13.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for name in (
        "gas_drive_eff",
        "entry_drive_eff",
        "gas_escape_eff",
        "wave_speed",
        "coupling_interval",
        "ds",
        "dz",
        "t_end",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    payload = json.dumps(run(args), ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
