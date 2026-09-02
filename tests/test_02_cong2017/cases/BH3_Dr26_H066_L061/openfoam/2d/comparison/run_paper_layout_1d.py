#!/usr/bin/env python3
"""Run the project 1D model for Cong et al. (2017) Campaign-2 B-H3.

This comparison run deliberately uses the axial dimensions drawn in paper Fig. 1(b),
which are also used by the neighbouring OpenFOAM case:

    tank -- 3.47 m -- riser -- 2.51 m -- valve -- 0.61 m -- cap

It is kept separate from the historical 63-case scan outputs so that the original
campaign results are not overwritten by this geometry-alignment run.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = CASE_ROOT / "model"
OUT = HERE / "model_1d"
sys.path.insert(0, str(MODEL_ROOT))

from cong2017_network_twofluid import NetworkCase, run_network  # noqa: E402


def first_crossing(t: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    idx = np.flatnonzero(y >= threshold)
    return None if idx.size == 0 else float(t[idx[0]])


def max_window_slope(t: np.ndarray, y: np.ndarray, start: float, stop: float, width: float = 0.6) -> float | None:
    mask = (t >= start) & (t <= stop) & np.isfinite(y)
    tt, yy = t[mask], y[mask]
    if len(tt) < 4:
        return None
    best = None
    left = 0
    for right in range(len(tt)):
        while tt[right] - tt[left] > width:
            left += 1
        if right - left >= 2 and tt[right] > tt[left]:
            slope = float((yy[right] - yy[left]) / (tt[right] - tt[left]))
            best = slope if best is None else max(best, slope)
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case = NetworkCase(
        D=0.050,
        Dr=0.026,
        riser_height=1.80,
        L_up=3.47,
        L_mid=2.51,
        L_down=0.61,
        x_riser_at=3.47,
        pocket_downstream=True,
        # The paper's H0=0.66 m is measured from the 50 mm pipe invert.
        # This model's vertical datum is the pipe crown, hence 0.66-0.05=0.61 m.
        reservoir_head=0.61,
        air_head=0.0,
        init_water_level=0.61,
        Hop_cap=10.0,
        x_transducer_at=6.44,
        valve_open_time=0.20,
        t_end=13.0,
    )

    started = time.time()
    rec = run_network(case, verbose=True)
    runtime = time.time() - started

    # Keep the full frame state needed by the browser comparison, without pickle.
    arrays = {
        "t": np.asarray(rec["t"], dtype=float),
        "wtop": np.asarray(rec["wtop"], dtype=float),
        "itop": np.asarray(rec["itop"], dtype=float),
        "pocket_head": np.asarray(rec["pocket_head"], dtype=float),
        "up_head": np.asarray(rec["up_head"], dtype=float),
        "frames_t": np.asarray(rec["frames_t"], dtype=float),
        "frames_alt": np.asarray(rec["frames_alt"], dtype=np.float32),
        "frames_alr": np.asarray(rec["frames_alr"], dtype=np.float32),
        "frames_agr": np.asarray(rec["frames_agr"], dtype=np.float32),
        "frames_itop": np.asarray(rec["frames_itop"], dtype=float),
        "frames_core_mass": np.asarray(rec["frames_core_mass"], dtype=float),
        "xt": np.asarray(rec["xt"], dtype=float),
        "zr": np.asarray(rec["zr"], dtype=float),
        "dx": np.asarray([rec["dx"]], dtype=float),
        "dz": np.asarray([rec["dz"]], dtype=float),
    }
    np.savez_compressed(OUT / "paper_layout_1d_frames.npz", **arrays)

    t = arrays["t"]
    wtop = arrays["wtop"]
    itop = arrays["itop"]
    ta = first_crossing(t, itop, 0.02)
    t_rim = first_crossing(t, wtop, 0.98 * case.riser_height)
    stop = t_rim if t_rim is not None else float(t[int(np.nanargmax(itop))])
    v_fs = None if ta is None else max_window_slope(t, wtop, ta, stop)
    v_int = None if ta is None else max_window_slope(t, itop, ta, stop)

    summary = {
        "schema_version": 1,
        "case": "BH3_Dr26_H066_L061",
        "paper_run": "B-H3",
        "model": "project 1D two-fluid network model",
        "comparison_role": "geometry-aligned model prediction; not an OpenFOAM validation result",
        "status": "COMPLETE" if float(t[-1]) >= 12.999 else "PARTIAL",
        "runtime_s": round(runtime, 2),
        "geometry_m": {
            "D": case.D,
            "Dr": case.Dr,
            "riser_height": case.riser_height,
            "tank_to_riser": case.L_up,
            "riser_to_valve": case.L_mid,
            "valve_to_cap_L0": case.L_down,
            "total_pipe": case.L_tunnel,
        },
        "initial_conditions": {
            "paper_H0_above_pipe_invert_m": 0.66,
            "model_head_above_pipe_crown_m": case.reservoir_head,
            "initial_riser_water_level_above_pipe_crown_m": case.init_water_level,
            "pocket_gauge_head": case.air_head,
            "valve_opening_duration_s": case.valve_open_time,
        },
        "dimensionless_air_volume": case.V_air / (0.25 * math.pi * case.Dr**2 * 0.66),
        "metrics": {
            "Ta_model_s": ta,
            "v_fs_model_m_per_s": v_fs,
            "v_int_model_m_per_s": v_int,
            "Yfs_max_m": float(np.nanmax(wtop)),
            "Yint_max_m": float(np.nanmax(itop)),
            "geyser_model": bool(t_rim is not None),
            "t_rim_model_s": t_rim,
        },
        "experiment_targets": {
            "Ta_s": 8.18,
            "v_fs_m_per_s": 0.657,
            "v_int_m_per_s": 0.916,
            "geyser_observed": True,
        },
        "artifacts": {"frames_npz": "paper_layout_1d_frames.npz"},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
