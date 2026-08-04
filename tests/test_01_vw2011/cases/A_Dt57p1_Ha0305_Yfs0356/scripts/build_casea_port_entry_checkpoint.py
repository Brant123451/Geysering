"""Build the Case-A pre-T checkpoint at the west edge of the real side port.

The earlier exploratory post-T checkpoint was attached to the grid face
nearest the tower centreline.  The 2-D geometry opens the tower over the
finite interval ``x_T +/- D_T/2``; the first topology event is therefore the
west edge of that opening.  This script reruns only the validated pre-arrival
shock-fit model and stops at the nearest finite-volume face to that edge.  It
does not create a tower state or alter any conserved field at the event.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = CASE_ROOT / "model"
OUTPUT_DIR = CASE_ROOT / "outputs"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import (  # noqa: E402
    CaseASideTShockFit,
    case_a_config,
)


TOWER_CENTRE_X = 3.516
TOWER_DIAMETER = 0.0571
PORT_WEST_X = TOWER_CENTRE_X - 0.5 * TOWER_DIAMETER


def build_checkpoint(*, cells: int, requested_dt: float, output: Path) -> Path:
    if cells < 20:
        raise ValueError("at least 20 horizontal cells are required")
    if requested_dt <= 0.0:
        raise ValueError("requested_dt must be positive")

    base = case_a_config(dx=4.006 / cells)
    solver = CaseASideTShockFit(replace(base, vent_x=PORT_WEST_X))
    state = solver.case_b_initial_state()
    initial_liquid_volume = float(
        np.sum(state.area, dtype=np.float64) * solver.dx
    )
    event_calls = 0
    while True:
        event_calls += 1
        advanced = solver.step_until_junction(
            state,
            requested_dt,
            location_tolerance=(
                128.0
                * np.finfo(float).eps
                * max(1.0, solver.junction_face_x)
            ),
        )
        state = advanced.state
        if advanced.reached:
            break
        if state.time > 30.0:
            raise RuntimeError("the fitted interface did not reach the side port")

    liquid_volume = float(np.sum(state.area, dtype=np.float64) * solver.dx)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=np.asarray([state.time]),
        area=np.asarray(state.area, dtype=float),
        discharge=np.asarray(state.discharge, dtype=float),
        interface_x=np.asarray([state.interface_x]),
        interface_speed=np.asarray([state.interface_speed]),
        interface_free_surface_depth=np.asarray(
            [state.interface_free_surface_depth]
        ),
        interface_free_surface_velocity=np.asarray(
            [state.interface_free_surface_velocity]
        ),
        interface_pressurised_head=np.asarray(
            [state.interface_pressurised_head]
        ),
        interface_pressurised_velocity=np.asarray(
            [state.interface_pressurised_velocity]
        ),
        interface_residual_linf=np.asarray([state.interface_residual_linf]),
        wetting_front_x=np.asarray([state.wetting_front_x]),
        gas_mass=np.asarray([state.gas.mass]),
        gas_volume=np.asarray([state.gas.volume]),
        gas_pressure=np.asarray([state.gas.pressure_abs]),
        air_pressure=np.asarray([state.air_pressure_abs]),
        dx=np.asarray([solver.dx]),
        x=np.asarray(solver.x, dtype=float),
        junction_face_index=np.asarray(
            [solver.junction_face_index], dtype=np.int32
        ),
        junction_face_x=np.asarray([solver.junction_face_x]),
        physical_port_west_x=np.asarray([PORT_WEST_X]),
        liquid_volume=np.asarray([liquid_volume]),
        initial_liquid_volume=np.asarray([initial_liquid_volume]),
        liquid_volume_error=np.asarray(
            [liquid_volume - initial_liquid_volume]
        ),
    )
    metadata = {
        "checkpoint": str(output.resolve()),
        "event_calls": event_calls,
        "time": float(state.time),
        "cells": int(solver.ncell),
        "dx": float(solver.dx),
        "physical_port_west_x": PORT_WEST_X,
        "discrete_event_face_x": float(solver.junction_face_x),
        "event_position_error": float(solver.junction_face_x - PORT_WEST_X),
        "interface_speed": float(state.interface_speed),
        "interface_residual_linf": float(state.interface_residual_linf),
        "liquid_volume": liquid_volume,
        "initial_liquid_volume": initial_liquid_volume,
        "liquid_volume_error": float(
            liquid_volume - initial_liquid_volume
        ),
        # Retained as an internal core diagnostic only.  The authoritative
        # conservation audit is the direct before/after inventory above.
        "core_cumulative_projection_diagnostic": float(
            state.cumulative_liquid_volume_residual
        ),
        "gas_mass": float(state.gas.mass),
        "gas_volume": float(state.gas.volume),
        "gas_pressure_abs": float(state.gas.pressure_abs),
        "nonlinear_converged": bool(state.nonlinear_converged),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "casea_port_west_event_dx40_checkpoint.npz",
    )
    args = parser.parse_args()
    print(build_checkpoint(cells=args.cells, requested_dt=args.dt, output=args.output))


if __name__ == "__main__":
    main()
