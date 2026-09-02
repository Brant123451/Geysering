"""Explore a post-impact topology switch for Case B's 1-D horizontal core.

The frozen solver is left untouched.  Before the fitted gas/liquid front
reaches the closed right wall, this script advances the established Tosan
shock-fit model.  Once the pressurised reach has vanished, it retires the
interface split and advances the entire horizontal conduit with the same
Saint-Venant finite-volume kernel and closed-wall boundaries.

This is sensitivity evidence only.  It tests whether the missing right-end
surface wave is caused by retaining a non-existent pressurised branch after
impact; it must not replace the manuscript baseline without a separate audit.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL = CASE_ROOT / "model" / "tosan2021_horizontal_shockfit.py"
BUILDER = Path(__file__).with_name("caseB_rebuild_1d_tosan2021.py")
OUTPUT_DIR = CASE_ROOT / "outputs" / "sensitivity_postimpact_wave_1d"
OUTPUT_JSON = OUTPUT_DIR / "postimpact_topology_metrics.json"
CACHED_SENSITIVITY = (
    CASE_ROOT / "outputs" / "sensitivity_wave_1d" / "wave_resolution_metrics.json"
)

LENGTH = 4.006
DIAMETER = 0.094
DX = 0.005
T_END = 8.95
OUTPUT_DT = 0.02
POST_IMPACT_CFL = 0.45


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot_with_depth(solver, state, *, phase: str) -> dict:
    raw = solver.snapshot(state)
    depth = solver.section.depth_from_area(
        np.clip(np.asarray(raw["area"], dtype=float), 0.0, solver.section.full_area)
    )
    return {
        "t": round(float(raw["time"]), 8),
        "phase": phase,
        "x": np.round(np.asarray(raw["x"], dtype=float), 7).tolist(),
        "h": np.round(np.asarray(depth, dtype=float), 8).tolist(),
        "area": np.round(np.asarray(raw["area"], dtype=float), 12).tolist(),
        "discharge": np.round(np.asarray(raw["discharge"], dtype=float), 12).tolist(),
        "interface_x": round(float(raw["interface_x"]), 8),
        "air_mass": float(raw["air_mass"]),
        "water_volume": float(raw["water_volume"]),
    }


def right_branch_wave_metric(frame: dict) -> dict | None:
    x = np.asarray(frame["x"], dtype=float)
    h = np.asarray(frame["h"], dtype=float)
    mask = (
        (x >= 3.56)
        & (x <= 3.98)
        & (h > 0.002)
        & (h < DIAMETER - 0.002)
    )
    xx, yy = x[mask], h[mask]
    if xx.size < 21:
        return None
    order = np.argsort(xx)
    xx, yy = xx[order], yy[order]
    dx = float(np.median(np.diff(xx)))
    window = max(5, int(round(0.10 / dx)))
    if window % 2 == 0:
        window += 1
    if window >= xx.size:
        return None
    half = window // 2
    valid = np.arange(half, xx.size - half)
    jump_edges = np.flatnonzero(np.abs(np.diff(yy)) > 0.002)
    for edge in jump_edges:
        valid = valid[np.abs(valid - edge) > half]
    if valid.size < 3:
        return None
    smooth = np.convolve(yy, np.ones(window) / window, mode="same")
    residual = yy - smooth
    crest = int(valid[np.argmax(residual[valid])])
    trough = int(valid[np.argmin(residual[valid])])
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


def postimpact_step(module, solver, state, dt: float, pressure_hook):
    wetdry = module.central_upwind_wet_dry_step(
        module.WetDryState(state.area, state.discharge),
        dx=solver.dx,
        dt=dt,
        section=solver.section,
        cfl=POST_IMPACT_CFL,
        dry_area_fraction=solver.config.dry_area_fraction,
        manning_n=solver.config.manning_n,
        darcy_friction=solver.config.darcy_friction,
        bed_slope=solver.config.bed_slope,
        left_boundary="wall",
        right_boundary="wall",
    )
    gas_volume = float(
        np.sum(
            solver.section.full_area
            - np.minimum(np.maximum(wetdry.area, 0.0), solver.section.full_area)
        )
        * solver.dx
    )
    minimum_volume = 1.0e-9 * solver.section.full_area * solver.config.length
    gas = state.gas.with_volume(max(gas_volume, minimum_volume))
    time_new = state.time + dt
    pressure = pressure_hook(time_new, solver.config.length, gas.pressure_abs)
    return replace(
        state,
        time=time_new,
        area=wetdry.area,
        discharge=wetdry.discharge,
        gas=gas,
        air_pressure_abs=pressure,
        interface_x=solver.config.length,
        interface_speed=0.0,
        wetting_front_x=solver._wetting_front(wetdry.area, state.wetting_front_x),
        vented=True,
    )


def restart_from_cached_impact(module, builder, vertical_frames: list[dict]):
    """Reconstruct an impact restart from the completed fine-grid run.

    Discharge follows directly from the discrete continuity equation using
    centred time differentiation and the known closed left-wall flux.  The
    tiny domain-integrated derivative caused by rounded cached depths is
    removed before integration so both wall fluxes remain zero.
    """
    config = module.HorizontalConfig(
        length=LENGTH,
        diameter=DIAMETER,
        valve_x=0.546,
        vent_x=3.516,
        dx=DX,
        wave_speed=100.0,
        gamma=1.4,
        initial_air_head=0.610,
        initial_water_head=0.356,
        wetting_front_report_fraction=builder.VISIBLE_WATER_AREA_FRACTION,
    )
    pressure_hook = builder._tower_pressure_hook(vertical_frames)
    solver = module.Tosan2021HorizontalShockFit(
        config,
        vent_pressure_hook=pressure_hook,
    )
    cached = json.loads(CACHED_SENSITIVITY.read_text(encoding="utf-8"))
    variant = min(cached["variants"], key=lambda row: abs(float(row["dx_m"]) - DX))
    source = variant["frames"]
    impact_index = next(
        index
        for index, row in enumerate(source)
        if float(row["interface_x"]) >= LENGTH - 0.5 * solver.dx
    )
    if impact_index <= 0 or impact_index >= len(source) - 1:
        raise RuntimeError("cached impact frame cannot support centred reconstruction")
    previous, current, following = source[impact_index - 1 : impact_index + 2]
    h_previous = np.asarray(previous["h"], dtype=float)
    h_current = np.asarray(current["h"], dtype=float)
    h_following = np.asarray(following["h"], dtype=float)
    area_previous = np.asarray(solver.section.area_from_depth(h_previous))
    area_current = np.asarray(solver.section.area_from_depth(h_current))
    area_following = np.asarray(solver.section.area_from_depth(h_following))
    time_span = float(following["t"]) - float(previous["t"])
    area_t = (area_following - area_previous) / time_span
    area_t -= np.mean(area_t)
    face_discharge = np.zeros(area_current.size + 1)
    face_discharge[1:] = -np.cumsum(area_t) * solver.dx
    discharge = 0.5 * (face_discharge[:-1] + face_discharge[1:])

    base_state = solver.case_b_initial_state()
    gas_volume = float(
        np.sum(solver.section.full_area - np.minimum(area_current, solver.section.full_area))
        * solver.dx
    )
    gas = base_state.gas.with_volume(gas_volume)
    impact_time = float(current["t"])
    pressure = pressure_hook(impact_time, LENGTH, gas.pressure_abs)
    state = replace(
        base_state,
        time=impact_time,
        area=area_current,
        discharge=discharge,
        gas=gas,
        air_pressure_abs=pressure,
        interface_x=LENGTH,
        interface_speed=0.0,
        wetting_front_x=solver._wetting_front(area_current, base_state.wetting_front_x),
        vented=True,
    )
    diagnostic = {
        "source_time_previous_s": float(previous["t"]),
        "source_time_impact_s": impact_time,
        "source_time_following_s": float(following["t"]),
        "reconstructed_left_wall_discharge_m3s": float(face_discharge[0]),
        "reconstructed_right_wall_discharge_m3s": float(face_discharge[-1]),
        "maximum_reconstructed_velocity_ms": float(
            np.max(
                np.abs(
                    np.divide(
                        discharge,
                        area_current,
                        out=np.zeros_like(discharge),
                        where=area_current > 1.0e-12,
                    )
                )
            )
        ),
    }
    return solver, pressure_hook, state, diagnostic


def run_candidate(module, builder, vertical_frames: list[dict]):
    solver, pressure_hook, state, restart_diagnostic = restart_from_cached_impact(
        module, builder, vertical_frames
    )
    frames = [
        snapshot_with_depth(solver, state, phase="whole_pipe_wall_reflection")
    ]
    next_output = state.time + OUTPUT_DT
    tolerance = 1.0e-12
    while state.time < T_END - tolerance:
        target = min(next_output, T_END)
        while state.time < target - tolerance:
            dry_area = solver.config.dry_area_fraction * solver.section.full_area
            velocity = np.divide(
                state.discharge,
                state.area,
                out=np.zeros_like(state.discharge),
                where=state.area > dry_area,
            )
            speed = float(
                np.max(np.abs(velocity) + solver.section.celerity(state.area))
            )
            stable = POST_IMPACT_CFL * solver.dx / max(speed, 1.0e-12)
            dt = min(0.999999 * stable, target - state.time)
            state = postimpact_step(module, solver, state, dt, pressure_hook)
        frames.append(
            snapshot_with_depth(solver, state, phase="whole_pipe_wall_reflection")
        )
        next_output += OUTPUT_DT
    return frames, float(frames[0]["t"]), restart_diagnostic


def main() -> None:
    module = load_module("caseb_postimpact_solver", MODEL)
    builder = load_module("caseb_postimpact_builder", BUILDER)
    times = np.arange(0.0, T_END + 0.5 * OUTPUT_DT, OUTPUT_DT).tolist()
    _, _, vertical_frames = builder._run_vertical_reference(times)
    print("Running post-impact topology-switch candidate", flush=True)
    frames, impact_time, restart_diagnostic = run_candidate(
        module, builder, vertical_frames
    )
    tracking = [right_branch_wave_metric(frame) for frame in frames]
    detected = [
        (frame, track)
        for frame, track in zip(frames, tracking)
        if track is not None and frame["t"] >= impact_time
    ]
    gas_mass = np.asarray([frame["air_mass"] for frame in frames])
    water_volume = np.asarray([frame["water_volume"] for frame in frames])
    payload = {
        "status": "exploratory_sensitivity_only",
        "governing_equations_modified": False,
        "baseline_solver_modified": False,
        "numerical_topology_modified": True,
        "postimpact_boundary": "whole horizontal domain; wall/wall",
        "dx_m": DX,
        "output_dt_s": OUTPUT_DT,
        "impact_time_s": impact_time,
        "restart_diagnostic": restart_diagnostic,
        "frames": frames,
        "tracking": tracking,
        "metrics": {
            "tracked_postimpact_frames": len(detected),
            "first_tracked_time_s": detected[0][0]["t"] if detected else None,
            "maximum_residual_peak_to_peak_m": max(
                (track["residual_peak_to_peak_m"] for _, track in detected),
                default=0.0,
            ),
            "gas_mass_relative_range": float(np.ptp(gas_mass) / gas_mass[0]),
            "water_volume_relative_range": float(
                np.ptp(water_volume) / water_volume[0]
            ),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"impact_time_s={impact_time:.6f}", flush=True)
    print(f"Output -> {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
