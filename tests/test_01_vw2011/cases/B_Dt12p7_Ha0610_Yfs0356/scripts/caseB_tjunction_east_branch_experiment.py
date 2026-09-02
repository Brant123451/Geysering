"""Run Case B with conservative gas continuation into the closed east leg.

The governing equations and all frozen parameters are unchanged.  The only
exploratory changes are the T-junction topology and the conservative
moving-front momentum handoff: after the horizontal crown current reaches the
tower junction, it may continue into the downstream closed branch while the
existing conservative gas exchange into the riser remains active.  The liquid
trace at the moving nose follows the Rankine-Hugoniot mass jump; no wave shape
or amplitude is prescribed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model" / "vw2011_network_twofluid.py"
OUTPUT_DIR = CASE_ROOT / "outputs" / "sensitivity_tjunction_east_branch"
OUTPUT_JSON = OUTPUT_DIR / "east_branch_metrics.json"


def load_model():
    spec = importlib.util.spec_from_file_location("caseb_tjunction_east_model", MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODEL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def right_wave_metric(x: np.ndarray, depth: np.ndarray, x0: float) -> dict | None:
    mask = (x >= x0 + 0.03) & (x <= 3.99)
    xx, yy = x[mask], depth[mask]
    if xx.size < 7:
        return None
    # Remove the branch-scale tilt; retain only a spatial crest/trough pair.
    trend = np.polyval(np.polyfit(xx, yy, 1), xx)
    residual = yy - trend
    crest = int(np.argmax(residual))
    trough = int(np.argmin(residual))
    amplitude = float(residual[crest] - residual[trough])
    if amplitude < 2.0e-4:
        return None
    return {
        "crest_x": float(xx[crest]),
        "crest_h": float(yy[crest]),
        "trough_x": float(xx[trough]),
        "trough_h": float(yy[trough]),
        "residual_peak_to_peak_m": amplitude,
    }


def main() -> None:
    model = load_model()
    case = model.NetworkCase(
        Dr=0.0127,
        air_head=0.610,
        init_water_level=0.356,
        t_end=8.35,
        allow_downstream_crown_front=True,
        enable_horizontal_gas_momentum_coupling=True,
    )
    print("Running two-fluid east-branch continuation sensitivity", flush=True)
    rec = model.run_network(case, verbose=True)
    x = np.asarray(rec["xt"], dtype=float)
    dx = float(rec["dx"])
    right = x >= case.x_riser
    frames = []
    wall_arrival = None
    for index, time in enumerate(rec["frames_t"]):
        alpha_l = np.asarray(rec["frames_alt"][index], dtype=float)
        alpha_g = np.clip(1.0 - alpha_l, 0.0, 1.0)
        depth = case.D * np.asarray(model._depth_frac(alpha_l), dtype=float)
        gas_cells = np.flatnonzero((alpha_g > 0.05) & right)
        front_x = float(x[gas_cells[-1]] + 0.5 * dx) if gas_cells.size else None
        wall_alpha = float(alpha_g[-1])
        if wall_arrival is None and wall_alpha > 0.05:
            wall_arrival = float(time)
        frames.append(
            {
                "t": float(time),
                "x": np.round(x, 7).tolist(),
                "h": np.round(depth, 8).tolist(),
                "alpha_g": np.round(alpha_g, 8).tolist(),
                "right_gas_volume_m3": float(
                    np.sum(alpha_g[right]) * case.A * dx
                ),
                "right_front_x": front_x,
                "right_wall_alpha_g": wall_alpha,
                "track": right_wave_metric(x, depth, case.x_riser),
            }
        )
    post_wall = [
        row for row in frames if wall_arrival is not None and row["t"] >= wall_arrival
    ]
    gas_mass = np.asarray(rec.get("tun_gas_mass", []), dtype=float)
    liquid = np.asarray(rec.get("tot_liq", []), dtype=float)
    payload = {
        "status": "exploratory_sensitivity_only",
        "governing_equations_modified": False,
        "frozen_defaults_modified": False,
        "junction_topology_modified": True,
        "allow_downstream_crown_front": True,
        "enable_horizontal_gas_momentum_coupling": True,
        "moving_front_liquid_trace": "rankine_hugoniot_one_cell_relaxation",
        "wall_arrival_time_s": wall_arrival,
        "metrics": {
            "maximum_right_gas_volume_m3": max(
                (row["right_gas_volume_m3"] for row in frames), default=0.0
            ),
            "maximum_wall_alpha_g": max(
                (row["right_wall_alpha_g"] for row in frames), default=0.0
            ),
            "maximum_postwall_wave_m": max(
                (
                    row["track"]["residual_peak_to_peak_m"]
                    for row in post_wall
                    if row["track"] is not None
                ),
                default=0.0,
            ),
            "tunnel_gas_mass_relative_range": (
                float(np.ptp(gas_mass) / gas_mass[0]) if gas_mass.size else None
            ),
            "total_liquid_relative_range": (
                float(np.ptp(liquid) / liquid[0]) if liquid.size else None
            ),
        },
        "frames": frames,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"wall_arrival_time_s={wall_arrival}", flush=True)
    print(f"Output -> {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
