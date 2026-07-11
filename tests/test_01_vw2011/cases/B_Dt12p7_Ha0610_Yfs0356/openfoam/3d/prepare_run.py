#!/usr/bin/env python3
"""Materialise small runtime includes and deterministic probe locations."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
SYSTEM = HERE / "system"
RUNTIME = HERE / "outputs" / "runtime"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def probe_lines(
    offsets: list[tuple[float, float]], y0: float, y1: float, step: float
) -> str:
    count = round((y1 - y0) / step)
    rows: list[str] = []
    for dx, dz in offsets:
        for index in range(count + 1):
            y = y0 + index * step
            rows.append(f"({3.516 + dx:.6f} {y:.6f} {dz:.6f})")
    return "\n".join(rows) + "\n"


def main() -> None:
    stage = os.environ.get("CASEB_STAGE", "full")
    defaults = {"hold": 1.00, "smoke": 0.50, "full": 10.50, "mesh": 0.0}
    if stage not in defaults:
        raise SystemExit(f"CASEB_STAGE must be one of {sorted(defaults)}")

    end_time = env_float("CASEB_END_TIME", defaults[stage])
    max_co = env_float("CASEB_MAX_CO", 0.30)
    max_alpha_co = env_float("CASEB_MAX_ALPHA_CO", 0.20)
    max_delta_t = env_float("CASEB_MAX_DELTA_T", 0.00025)
    write_interval = env_float("CASEB_WRITE_INTERVAL", 0.10)
    c_alpha = env_float("CASEB_C_ALPHA", 1.0)
    initial_air_head = env_float("CASEB_HA0", 0.610)
    valve_mode = os.environ.get(
        "CASEB_VALVE_MODE", "closed" if stage == "hold" else "opening"
    )
    valve_open_time = env_float("CASEB_VALVE_OPEN_TIME", 0.25)
    if valve_mode not in {"opening", "closed", "instant"}:
        raise SystemExit("CASEB_VALVE_MODE must be opening, closed, or instant")
    if stage == "hold" and valve_mode != "closed":
        raise SystemExit("CASEB_STAGE=hold requires CASEB_VALVE_MODE=closed")
    gas_eos = os.environ.get("CASEB_GAS_EOS", "perfectGas")
    if gas_eos not in {"perfectGas", "rhoConst"}:
        raise SystemExit("CASEB_GAS_EOS must be perfectGas or rhoConst")
    values = (
        end_time,
        max_co,
        max_alpha_co,
        max_delta_t,
        write_interval,
        c_alpha,
        initial_air_head,
    )
    if not all(math.isfinite(value) for value in (*values, valve_open_time)):
        raise SystemExit("Runtime controls must be finite")
    positive = (max_co, max_alpha_co, max_delta_t, write_interval, c_alpha)
    if any(value <= 0 for value in positive) or initial_air_head <= 0:
        raise SystemExit("Courant, timestep, output, cAlpha and head must be positive")
    if end_time < 0 or (stage != "mesh" and end_time <= 0):
        raise SystemExit("endTime must be positive for solver stages")
    if valve_open_time < 0:
        raise SystemExit("CASEB_VALVE_OPEN_TIME cannot be negative")
    if stage != "mesh":
        write_interval = min(write_interval, end_time)
        probe_interval = min(0.005, end_time)
        plume_interval = min(0.010, end_time)
        accounting_interval = min(0.010, end_time)
    else:
        probe_interval = 0.005
        plume_interval = 0.010
        accounting_interval = 0.010

    (SYSTEM / "runControl").write_text(
        "\n".join(
            (
                f"endTime         {end_time:.10g};",
                "writeControl    adjustableRunTime;",
                f"writeInterval   {write_interval:.10g};",
                f"maxCo           {max_co:.10g};",
                f"maxAlphaCo      {max_alpha_co:.10g};",
                f"maxDeltaT       {max_delta_t:.10g};",
                f"caseBProbeInterval      {probe_interval:.10g};",
                f"caseBPlumeInterval      {plume_interval:.10g};",
                f"caseBAccountingInterval {accounting_interval:.10g};",
                "",
            )
        )
    )
    (SYSTEM / "runSettings").write_text(f"cAlpha          {c_alpha:.10g};\n")
    initial_air_pressure = 101325.0 + 998.2 * 9.81 * initial_air_head
    set_fields = (SYSTEM / "setFieldsDict").read_text()
    if set_fields.count("107298.329") != 2:
        raise SystemExit("Expected two baseline pressure tokens in setFieldsDict")
    set_fields = set_fields.replace("107298.329", f"{initial_air_pressure:.6f}")
    (SYSTEM / "setFieldsDict.runtime").write_text(set_fields)
    set_expr = (SYSTEM / "setExprFieldsDict").read_text()
    if set_expr.count("107298.329") != 1:
        raise SystemExit("Expected one baseline pressure token in setExprFieldsDict")
    set_expr = set_expr.replace("107298.329", f"{initial_air_pressure:.6f}")
    (SYSTEM / "setExprFieldsDict.runtime").write_text(set_expr)
    (HERE / "constant" / "airThermo").write_text(
        (HERE / "constant" / f"airThermo.{gas_eos}").read_text()
    )
    offsets = [(0, 0), (0.003, 0), (-0.003, 0), (0, 0.003), (0, -0.003)]
    (SYSTEM / "towerProbeLocations").write_text(
        probe_lines(offsets, 0.052, 0.652, 0.005)
    )
    plume_offsets = [(0, 0), (0.004, 0), (-0.004, 0), (0, 0.004), (0, -0.004)]
    (SYSTEM / "plumeProbeLocations").write_text(
        probe_lines(plume_offsets, 0.662, 1.852, 0.010)
    )

    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "end_time_s": end_time,
        "mesh_preset": os.environ.get("CASEB_MESH", "base"),
        "valve_mode": valve_mode,
        "valve_open_time_s": valve_open_time,
        "initial_air_head_m": initial_air_head,
        "initial_air_absolute_pressure_Pa": initial_air_pressure,
        "gas_equation_of_state": gas_eos,
        "max_co": max_co,
        "max_alpha_co": max_alpha_co,
        "max_delta_t_s": max_delta_t,
        "field_write_interval_s": write_interval,
        "probe_write_interval_s": probe_interval,
        "plume_write_interval_s": plume_interval,
        "accounting_interval_s": accounting_interval,
        "c_alpha": c_alpha,
        "tower_probe_lines": 5,
        "tower_probe_spacing_m": 0.005,
        "plume_probe_lines": 5,
        "plume_probe_spacing_m": 0.010,
    }
    (RUNTIME / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
