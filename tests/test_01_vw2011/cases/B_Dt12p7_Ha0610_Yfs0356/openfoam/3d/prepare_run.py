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
TWOPHASEFLOW_COMMIT = "de9826f9ffb24f4b635ac97fd388ebd560cfc174"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SystemExit(f"{name} must be true or false")


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
    max_capillary_num = env_float("CASEB_MAX_CAPILLARY_NUM", 1.0)
    max_delta_t = env_float("CASEB_MAX_DELTA_T", 0.00025)
    write_interval = env_float("CASEB_WRITE_INTERVAL", 0.10)
    c_alpha = env_float("CASEB_C_ALPHA", 1.0)
    advection_scheme = os.environ.get(
        "CASEB_ADVECTION_SCHEME", "isoAdvection"
    )
    reconstruction_scheme = os.environ.get(
        "CASEB_RECONSTRUCTION_SCHEME", "plicRDF"
    )
    interpolate_normal = env_bool("CASEB_INTERPOLATE_NORMAL", False)
    curvature_model = os.environ.get("CASEB_CURVATURE_MODEL", "RDF")
    n_alpha_bounds = env_int("CASEB_N_ALPHA_BOUNDS", 5)
    n_alpha_corr = env_int("CASEB_N_ALPHA_CORR", 1)
    n_alpha_subcycles = env_int("CASEB_N_ALPHA_SUBCYCLES", 2)
    n_outer_correctors = env_int("CASEB_N_OUTER_CORRECTORS", 1)
    n_pressure_correctors = env_int("CASEB_N_CORRECTORS", 2)
    n_non_orthogonal_correctors = env_int(
        "CASEB_N_NON_ORTHOGONAL_CORRECTORS", 0
    )
    alpha_clip = env_bool("CASEB_ALPHA_CLIP", False)
    initial_air_head = env_float("CASEB_HA0", 0.610)
    valve_mode = os.environ.get(
        "CASEB_VALVE_MODE", "closed" if stage == "hold" else "opening"
    )
    valve_open_time = env_float("CASEB_VALVE_OPEN_TIME", 0.25)
    valve_seal_speed = env_float("CASEB_VALVE_SEAL_SPEED", 1.0)
    if valve_mode not in {"opening", "closed", "instant"}:
        raise SystemExit("CASEB_VALVE_MODE must be opening, closed, or instant")
    if stage == "hold" and valve_mode != "closed":
        raise SystemExit("CASEB_STAGE=hold requires CASEB_VALVE_MODE=closed")
    valve_representation = (
        "conformalNoSlipBaffle"
        if valve_mode == "closed"
        else "dissipativeResistance"
    )
    time_control = "runTime"
    gas_eos = os.environ.get("CASEB_GAS_EOS", "perfectGas")
    if gas_eos not in {"perfectGas", "rhoConst"}:
        raise SystemExit("CASEB_GAS_EOS must be perfectGas or rhoConst")
    if advection_scheme not in {"isoAdvection", "MULESScheme"}:
        raise SystemExit(
            "CASEB_ADVECTION_SCHEME must be isoAdvection or MULESScheme"
        )
    if reconstruction_scheme not in {"plicRDF", "isoAlpha", "gradAlpha"}:
        raise SystemExit(
            "CASEB_RECONSTRUCTION_SCHEME must be plicRDF, isoAlpha, or gradAlpha"
        )
    if curvature_model not in {"RDF", "fitParaboloid", "gradAlpha"}:
        raise SystemExit(
            "CASEB_CURVATURE_MODEL must be RDF, fitParaboloid, or gradAlpha"
        )
    if "CASEB_ALPHA_SMOOTH_CURVATURE" in os.environ:
        raise SystemExit(
            "CASEB_ALPHA_SMOOTH_CURVATURE belongs to stock "
            "compressibleInterFoam and is not valid for compressibleInterFlow"
        )
    if (
        advection_scheme != "MULESScheme"
        and "CASEB_C_ALPHA" in os.environ
        and not math.isclose(c_alpha, 1.0)
    ):
        raise SystemExit(
            "CASEB_C_ALPHA only changes interface compression when "
            "CASEB_ADVECTION_SCHEME=MULESScheme"
        )
    values = (
        end_time,
        max_co,
        max_alpha_co,
        max_capillary_num,
        max_delta_t,
        write_interval,
        c_alpha,
        initial_air_head,
        valve_seal_speed,
    )
    if not all(math.isfinite(value) for value in (*values, valve_open_time)):
        raise SystemExit("Runtime controls must be finite")
    positive = (
        max_co,
        max_alpha_co,
        max_capillary_num,
        max_delta_t,
        write_interval,
        c_alpha,
        valve_seal_speed,
    )
    if any(value <= 0 for value in positive) or initial_air_head <= 0:
        raise SystemExit(
            "Courant, timestep, output, cAlpha, seal speed and head must be positive"
        )
    if end_time < 0 or (stage != "mesh" and end_time <= 0):
        raise SystemExit("endTime must be positive for solver stages")
    if valve_open_time < 0:
        raise SystemExit("CASEB_VALVE_OPEN_TIME cannot be negative")
    if n_alpha_bounds < 1:
        raise SystemExit("CASEB_N_ALPHA_BOUNDS must be positive")
    positive_correctors = {
        "CASEB_N_ALPHA_CORR": n_alpha_corr,
        "CASEB_N_ALPHA_SUBCYCLES": n_alpha_subcycles,
        "CASEB_N_OUTER_CORRECTORS": n_outer_correctors,
        "CASEB_N_CORRECTORS": n_pressure_correctors,
    }
    invalid_correctors = [
        name for name, value in positive_correctors.items() if value < 1
    ]
    if invalid_correctors:
        raise SystemExit(
            f"{', '.join(invalid_correctors)} must be positive"
        )
    if n_non_orthogonal_correctors < 0:
        raise SystemExit(
            "CASEB_N_NON_ORTHOGONAL_CORRECTORS cannot be negative"
        )
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
                f"writeControl    {time_control};",
                f"writeInterval   {write_interval:.10g};",
                f"maxCo           {max_co:.10g};",
                f"maxAlphaCo      {max_alpha_co:.10g};",
                f"maxCapillaryNum {max_capillary_num:.10g};",
                f"maxDeltaT       {max_delta_t:.10g};",
                f"caseBProbeInterval      {probe_interval:.10g};",
                f"caseBPlumeInterval      {plume_interval:.10g};",
                f"caseBAccountingInterval {accounting_interval:.10g};",
                "",
            )
        )
    )
    (SYSTEM / "runSettings").write_text(
        f"cAlpha                  {c_alpha:.10g};\n"
        f"advectionScheme         {advection_scheme};\n"
        f"reconstructionScheme    {reconstruction_scheme};\n"
        f"interpolateNormal       {str(interpolate_normal).lower()};\n"
        f"nAlphaBounds            {n_alpha_bounds};\n"
        f"clip                    {str(alpha_clip).lower()};\n"
        "snapTol                 0;\n"
    )
    (SYSTEM / "alphaCorrectors").write_text(
        f"nAlphaCorr      {n_alpha_corr};\n"
        f"nAlphaSubCycles {n_alpha_subcycles};\n"
    )
    (SYSTEM / "pimpleCorrectors").write_text(
        f"nOuterCorrectors         {n_outer_correctors};\n"
        f"nCorrectors              {n_pressure_correctors};\n"
        f"nNonOrthogonalCorrectors {n_non_orthogonal_correctors};\n"
    )
    (HERE / "constant" / "surfaceForces").write_text(
        "surfaceForces\n"
        "{\n"
        "    sigma                       0.072;\n"
        f"    surfaceTensionForceModel    {curvature_model};\n"
        "    curvFromTr                  true;\n"
        "    accelerationForceModel      gravity;\n"
        "    deltaFunctionModel          alphaCSF;\n"
        "}\n"
    )
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
        "valve_representation": valve_representation,
        "valve_open_time_s": valve_open_time,
        "valve_seal_speed_m_per_s": valve_seal_speed,
        "initial_air_head_m": initial_air_head,
        "initial_air_absolute_pressure_Pa": initial_air_pressure,
        "gas_equation_of_state": gas_eos,
        "solver": "compressibleInterFlow",
        "two_phase_flow_commit": TWOPHASEFLOW_COMMIT,
        "advection_scheme": advection_scheme,
        "reconstruction_scheme": reconstruction_scheme,
        "interpolate_normal": interpolate_normal,
        "curvature_model": curvature_model,
        "max_co": max_co,
        "max_alpha_co": max_alpha_co,
        "max_capillary_num": max_capillary_num,
        "max_delta_t_s": max_delta_t,
        "time_control": time_control,
        "field_write_interval_s": write_interval,
        "probe_write_interval_s": probe_interval,
        "plume_write_interval_s": plume_interval,
        "accounting_interval_s": accounting_interval,
        "c_alpha": c_alpha,
        "n_alpha_bounds": n_alpha_bounds,
        "n_alpha_corr": n_alpha_corr,
        "n_alpha_subcycles": n_alpha_subcycles,
        "n_outer_correctors": n_outer_correctors,
        "n_pressure_correctors": n_pressure_correctors,
        "n_non_orthogonal_correctors": n_non_orthogonal_correctors,
        "alpha_clip": alpha_clip,
        "tower_probe_lines": 5,
        "tower_probe_spacing_m": 0.005,
        "plume_probe_lines": 5,
        "plume_probe_spacing_m": 0.010,
    }
    (RUNTIME / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
