"""Run the standalone Case-A post-T graph from the exact event checkpoint.

This diagnostic runner is deliberately outside ``model``.  It reads the
bit-preserving shock-fit event checkpoint, constructs one conservative
physical post-T state, and advances the unified liquid/gas SSP-RK2 stage.  It
does not call the legacy Case-A main loop, OpenFOAM, or any plotting code.

The default vertical discretisation is the physical free-surface topology:
17 full cells plus one cut cell for Yfs=0.356 m, followed by a dry atmospheric
suffix.  ``legacy_uniform17`` is retained only to reproduce the rejected
initialisation that produced an internal vacuum and a node-pressure failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parent
MODEL_DIR = CASE_DIR / "model"
OUTPUT_DIR = CASE_DIR / "outputs"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_coupled_gas_network import CoupledGasParameters  # noqa: E402
from casea_horizontal_liquid_operator import (  # noqa: E402
    HorizontalLiquidParameters,
)
from casea_post_t_coupled_stage import (  # noqa: E402
    PostTCoupledGeometry,
    PostTCoupledState,
    advance_post_t_coupled_ssprk2,
)
from casea_post_t_liquid_stage import PostTLiquidGeometry  # noqa: E402
from casea_post_t_physical_closure import (  # noqa: E402
    CaseAPostTClosureParameters,
    CaseAPostTPhysicalClosure,
)
from casea_post_t_sideport_liquid_stage import (  # noqa: E402
    PostTSidePortGeometry,
)
from casea_topology_event import BranchFrontTopology  # noqa: E402


D_HORIZONTAL = 0.094
D_RISER = 0.0571
RISER_HEIGHT = 0.610
INITIAL_RISER_LEVEL = 0.356
LIQUID_WAVE_SPEED = 28.0
LIQUID_TENSION_HEAD = 0.05
VERTICAL_LOSS_COEFFICIENT = 0.75
RISER_CELLS = 30
POST_T_CFL = 0.20


def _initial_vertical_liquid(
    *, area_full: float, dz: float, mode: str
) -> tuple[np.ndarray, int]:
    area = np.zeros(RISER_CELLS, dtype=float)
    if mode == "physical18":
        active = int(math.ceil(INITIAL_RISER_LEVEL / dz))
        area[: active - 1] = area_full
        area[active - 1] = (
            area_full
            * (INITIAL_RISER_LEVEL - (active - 1) * dz)
            / dz
        )
    elif mode == "legacy_uniform17":
        active = int(math.floor(INITIAL_RISER_LEVEL / dz))
        area[:active] = (
            area_full * INITIAL_RISER_LEVEL / (active * dz)
        )
    else:  # pragma: no cover - argparse supplies the choices
        raise ValueError(f"unknown vertical initialisation {mode!r}")
    return area, active


def build_post_t_case(
    checkpoint: Path,
    *,
    coupling: str = "sideport",
    vertical_initialisation: str = "physical18",
) -> tuple[
    PostTCoupledState,
    PostTCoupledGeometry,
    CaseAPostTPhysicalClosure,
    float,
    dict[str, float | int | str],
]:
    """Map the exact event checkpoint to one post-T conservative state."""

    with np.load(checkpoint) as saved:
        event_time = float(saved["time"][0])
        horizontal_area = np.asarray(saved["area"], dtype=float).copy()
        horizontal_discharge = np.asarray(
            saved["discharge"], dtype=float
        ).copy()
        x = np.asarray(saved["x"], dtype=float)
        dx = float(saved["dx"][0])
        junction_face = int(saved["junction_face_index"][0])
        interface_x = float(saved["interface_x"][0])
        interface_speed = float(saved["interface_speed"][0])
        fitted_gas_mass = float(saved["gas_mass"][0])
        fitted_gas_volume = float(saved["gas_volume"][0])

    gas = CoupledGasParameters(
        horizontal_diameter=D_HORIZONTAL,
        vertical_diameter=D_RISER,
        rho_l=998.0,
        gas_temperature=293.0,
        vertical_gas_core_area_fraction=0.80,
    )
    area_horizontal = gas.horizontal_area
    area_riser = gas.vertical_area
    dz = RISER_HEIGHT / RISER_CELLS

    # Only the void connected to the fitted west pocket receives its mass.
    # Elastic rarefaction east of the fitted interface is liquid storage, not
    # a disconnected gas receiver.  The final scaling is exactly conservative.
    connected_void_volume = np.where(
        x < interface_x,
        np.maximum(
            area_horizontal
            - np.clip(horizontal_area, 0.0, area_horizontal),
            0.0,
        )
        * dx,
        0.0,
    )
    resolved_void = float(np.sum(connected_void_volume, dtype=np.float64))
    if resolved_void <= 0.0:
        raise FloatingPointError("event state has no connected horizontal void")
    horizontal_gas_mass = (
        fitted_gas_mass * connected_void_volume / resolved_void
    )
    gas_velocity = interface_speed * np.clip(
        x / max(interface_x, 0.5 * dx), 0.0, 1.0
    )
    horizontal_gas_momentum = horizontal_gas_mass * gas_velocity

    vertical_area, active_count = _initial_vertical_liquid(
        area_full=area_riser,
        dz=dz,
        mode=vertical_initialisation,
    )
    vertical_discharge = np.zeros(RISER_CELLS, dtype=float)
    vertical_gas_mass = np.zeros(RISER_CELLS, dtype=float)
    # Wet cells initially contain no material gas.  The dry suffix contains
    # the ambient gas already present in the open riser and is not tracer gas.
    vertical_gas_mass[active_count:] = (
        gas.rho_atmospheric * area_riser * dz
    )
    vertical_gas_momentum = np.zeros(RISER_CELLS, dtype=float)
    vertical_tracer_mass = np.zeros(RISER_CELLS, dtype=float)

    common = {
        "wave_speed": LIQUID_WAVE_SPEED,
        "gravity": gas.gravity,
        "rho_liquid": gas.rho_l,
        "gas_constant": gas.gas_constant,
        "gas_temperature": gas.gas_temperature,
        "atmospheric_pressure": gas.atmospheric_pressure,
        "tension_head": LIQUID_TENSION_HEAD,
    }
    horizontal_liquid = HorizontalLiquidParameters(
        area_full=area_horizontal,
        diameter=D_HORIZONTAL,
        cell_width=dx,
        **common,
    )
    vertical_liquid = HorizontalLiquidParameters(
        area_full=area_riser,
        diameter=D_RISER,
        cell_width=dz,
        **common,
    )
    closure = CaseAPostTPhysicalClosure(
        CaseAPostTClosureParameters(
            horizontal=horizontal_liquid,
            vertical=vertical_liquid,
            vertical_cell_width=dz,
        )
    )

    if coupling == "zero_node":
        liquid_geometry = PostTLiquidGeometry(
            junction_face_index=junction_face,
            horizontal_cell_width=dx,
            vertical_cell_width=dz,
            liquid_density=gas.rho_l,
            vertical_loss_coefficient=VERTICAL_LOSS_COEFFICIENT,
            atmospheric_pressure_abs=gas.atmospheric_pressure,
        )
    elif coupling == "sideport":
        liquid_geometry = PostTSidePortGeometry(
            horizontal_cell_width=dx,
            vertical_cell_width=dz,
            liquid_density=gas.rho_l,
            junction_center_x=3.516,
            opening_diameter=D_RISER,
            vertical_loss_coefficient=VERTICAL_LOSS_COEFFICIENT,
            atmospheric_pressure_abs=gas.atmospheric_pressure,
        )
    else:  # pragma: no cover - argparse supplies the choices
        raise ValueError(f"unknown coupling {coupling!r}")

    geometry = PostTCoupledGeometry(
        liquid=liquid_geometry,
        gas=gas,
        gas_junction_index=junction_face - 1,
        vertical_liquid_active_count=active_count,
    )
    state = PostTCoupledState(
        Alt=horizontal_area,
        Qlt=horizontal_discharge,
        Alr=vertical_area,
        Qlr=vertical_discharge,
        Mgt=horizontal_gas_mass,
        Jgt=horizontal_gas_momentum,
        Mgr=vertical_gas_mass,
        Jgrs=vertical_gas_momentum,
        Mgrs=vertical_tracer_mass,
        east_front=BranchFrontTopology("east", 0.0),
        vertical_front=BranchFrontTopology("vertical", 0.0),
    )
    dt = POST_T_CFL * min(dx, dz) / gas.sound_speed
    metadata: dict[str, float | int | str] = {
        "event_time": event_time,
        "coupling": coupling,
        "vertical_initialisation": vertical_initialisation,
        "dx": dx,
        "dz": dz,
        "horizontal_cells": horizontal_area.size,
        "vertical_cells": RISER_CELLS,
        "vertical_active_count": active_count,
        "junction_face_index": junction_face,
        "gas_junction_index": junction_face - 1,
        "fitted_gas_mass": fitted_gas_mass,
        "mapped_horizontal_gas_mass": float(
            np.sum(horizontal_gas_mass, dtype=np.float64)
        ),
        "fitted_gas_volume": fitted_gas_volume,
        "mapped_connected_void_volume": resolved_void,
        "initial_vertical_equivalent_height": float(
            np.sum(vertical_area, dtype=np.float64) * dz / area_riser
        ),
        "dt": dt,
    }
    return state, geometry, closure, dt, metadata


def _totals(state: PostTCoupledState, *, dx: float, dz: float) -> tuple[float, float, float]:
    liquid = float(np.sum(state.Alt) * dx + np.sum(state.Alr) * dz)
    gas = float(np.sum(state.Mgt) + np.sum(state.Mgr))
    tracer = float(np.sum(state.Mgt) + np.sum(state.Mgrs))
    return liquid, gas, tracer


def run(args: argparse.Namespace) -> Path:
    state, geometry, closure, nominal_dt, metadata = build_post_t_case(
        args.checkpoint,
        coupling=args.coupling,
        vertical_initialisation=args.vertical_initialisation,
    )
    time_value = float(metadata["event_time"])
    dx = float(metadata["dx"])
    dz = float(metadata["dz"])
    initial_liquid, initial_gas, initial_tracer = _totals(
        state, dx=dx, dz=dz
    )
    atmosphere = 0.0
    escaped_tracer = 0.0
    accepted = 0
    rejected = 0
    dt = nominal_dt
    minimum_dt = nominal_dt / 4096.0

    while time_value < args.end_time:
        step = min(dt, args.end_time - time_value)
        try:
            advanced = advance_post_t_coupled_ssprk2(
                state,
                dt=step,
                geometry=geometry,
                pressure_callback=closure,
                top_open=True,
            )
        except (FloatingPointError, ValueError):
            if not args.adaptive:
                raise
            rejected += 1
            dt *= 0.5
            if dt < minimum_dt:
                raise
            continue
        state = advanced.state
        time_value += step
        accepted += 1
        atmosphere += advanced.atmospheric_mass_exchange
        escaped_tracer += advanced.escaped_tracer_mass
        if args.adaptive and dt < nominal_dt:
            dt = min(nominal_dt, 1.05 * dt)

    final_liquid, final_gas, final_tracer = _totals(
        state, dx=dx, dz=dz
    )
    result = {
        **metadata,
        "end_time": time_value,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "final_dt": dt,
        "liquid_volume_error": final_liquid - initial_liquid,
        "gas_mass_error": final_gas - initial_gas + atmosphere,
        "tracer_mass_error": (
            final_tracer - initial_tracer + escaped_tracer
        ),
        "atmospheric_mass_exchange": atmosphere,
        "escaped_tracer_mass": escaped_tracer,
        "final_east_front": state.east_front.position,
        "final_vertical_front": state.vertical_front.position,
        "final_vertical_equivalent_height": float(
            np.sum(state.Alr) * dz / geometry.gas.vertical_area
        ),
        "minimum_horizontal_area_fraction": float(
            np.min(state.Alt) / geometry.gas.horizontal_area
        ),
        "minimum_active_vertical_area_fraction": float(
            np.min(state.Alr[: geometry.vertical_liquid_active_count])
            / geometry.gas.vertical_area
        ),
    }
    tag = (
        f"postt_{args.coupling}_{args.vertical_initialisation}_"
        f"to_{args.end_time:.4f}s"
    ).replace(".", "p")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / f"{tag}.npz",
        time=np.asarray([time_value]),
        Alt=state.Alt,
        Qlt=state.Qlt,
        Alr=state.Alr,
        Qlr=state.Qlr,
        Mgt=state.Mgt,
        Jgt=state.Jgt,
        Mgr=state.Mgr,
        Jgrs=state.Jgrs,
        Mgrs=state.Mgrs,
        east_front=np.asarray([state.east_front.position]),
        vertical_front=np.asarray([state.vertical_front.position]),
    )
    json_path = OUTPUT_DIR / f"{tag}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=OUTPUT_DIR / "postt_exact_event_dx40_checkpoint.npz",
    )
    parser.add_argument(
        "--coupling", choices=("zero_node", "sideport"), default="sideport"
    )
    parser.add_argument(
        "--vertical-initialisation",
        choices=("physical18", "legacy_uniform17"),
        default="physical18",
    )
    parser.add_argument("--end-time", type=float, default=6.50)
    parser.add_argument("--adaptive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
