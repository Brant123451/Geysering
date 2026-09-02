"""Regression tests for the face-aligned Case-A side-T handoff."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_coupled_gas_network import (  # noqa: E402
    CoupledGasParameters,
    advance_coupled_gas_network,
)
from casea_face_aligned_t import face_aligned_t_indices  # noqa: E402
from casea_shockfit_network import build_case_a_shockfit_solver  # noqa: E402
from vw2011_network_twofluid import NetworkCase, run_network  # noqa: E402


def _network_fixture(*, open_vertical_base: bool):
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
    cells = 46
    dx = 4.006 / cells
    dz = 0.02
    indices = face_aligned_t_indices(3.516, dx, cells)
    horizontal_liquid_area = np.full(cells, params.horizontal_area)
    horizontal_liquid_area[:indices.face] = 0.45 * params.horizontal_area
    horizontal_void = params.horizontal_area - horizontal_liquid_area
    horizontal_mass = np.zeros(cells)
    horizontal_mass[:indices.face] = (
        1.04
        * params.rho_atmospheric
        * horizontal_void[:indices.face]
        * dx
    )

    vertical_cells = 31
    vertical_liquid_area = np.full(vertical_cells, params.vertical_area)
    if open_vertical_base:
        vertical_liquid_area[0] = 0.50 * params.vertical_area
    vertical_void = np.maximum(
        params.vertical_area - vertical_liquid_area,
        params.void_floor_fraction * params.vertical_area,
    )
    vertical_mass = params.rho_atmospheric * vertical_void * dz
    return (
        params,
        indices,
        dx,
        dz,
        horizontal_mass,
        np.zeros(cells),
        horizontal_liquid_area,
        np.zeros(cells),
        vertical_mass,
        np.zeros(vertical_cells),
        np.zeros(vertical_cells),
        vertical_liquid_area,
        np.zeros(vertical_cells),
    )


def test_case_a_grid_and_shockfit_use_the_same_internal_face() -> None:
    cells = 46
    dx = 4.006 / cells
    indices = face_aligned_t_indices(3.516, dx, cells)
    solver = build_case_a_shockfit_solver(dx=dx)

    assert indices.face == 40
    assert indices.west_cell == 39
    assert indices.east_cell == 40
    assert indices.face_x == 40 * dx
    assert abs(indices.face_x - 3.516) <= 0.5 * dx
    assert solver.ncell == cells
    assert solver.junction_face_index == indices.face
    assert solver.junction_face_x == indices.face_x


def test_network_uses_face_indices_without_prearrival_vertical_gas() -> None:
    cells = 46
    dx = 4.006 / cells
    indices = face_aligned_t_indices(3.516, dx, cells)
    record = run_network(
        NetworkCase(
            Dr=0.0571,
            air_head=0.305,
            init_water_level=0.356,
            ds=dx,
            dz=0.02,
            t_end=0.002,
        ),
        verbose=False,
        output_interval=0.001,
    )

    assert record["junction_face"] == indices.face
    assert record["junction_west_cell"] == indices.west_cell
    assert record["junction_east_cell"] == indices.east_cell
    assert record["junction_face_x"] == indices.face_x
    assert max(float(np.sum(frame)) for frame in record["frames_mgrs"]) == 0.0


def test_liquid_full_riser_cannot_receive_seed_gas_from_west_pocket() -> None:
    fixture = _network_fixture(open_vertical_base=False)
    params, indices, dx, dz = fixture[:4]
    horizontal_mass_before = float(np.sum(fixture[4]))
    vertical_tracer_before = float(np.sum(fixture[10]))
    result = advance_coupled_gas_network(
        fixture[4],
        fixture[5],
        fixture[8],
        fixture[9],
        fixture[10],
        fixture[6],
        fixture[7],
        fixture[11],
        fixture[12],
        dx=dx,
        dz=dz,
        dt=1.0e-5,
        junction_index=indices.west_cell,
        params=params,
        horizontal_downstream_front_position=indices.face_x,
        horizontal_downstream_topology_front_position=indices.face_x,
        prefer_vertical_branch=True,
    )

    assert result.junction_mouth_area == 0.0
    assert result.junction_mass_transfer == 0.0
    assert float(np.sum(result.vertical_tracer_mass)) == vertical_tracer_before
    assert math.isclose(
        float(np.sum(result.horizontal_mass)),
        horizontal_mass_before,
        rel_tol=0.0,
        abs_tol=2.0e-14,
    )
    assert abs(result.total_mass_error) < 2.0e-13
    assert abs(result.tracer_mass_error) < 2.0e-13


def test_open_base_conservatively_receives_mass_from_west_pocket() -> None:
    fixture = _network_fixture(open_vertical_base=True)
    params, indices, dx, dz = fixture[:4]
    horizontal_mass_before = float(np.sum(fixture[4]))
    vertical_mass_before = float(np.sum(fixture[8]))
    pocket_pressure = 1.04 * params.atmospheric_pressure
    capillary_entry_pressure = 4.0 * params.surface_tension / 0.0571
    assert pocket_pressure > params.atmospheric_pressure + capillary_entry_pressure

    result = advance_coupled_gas_network(
        fixture[4],
        fixture[5],
        fixture[8],
        fixture[9],
        fixture[10],
        fixture[6],
        fixture[7],
        fixture[11],
        fixture[12],
        dx=dx,
        dz=dz,
        dt=1.0e-5,
        junction_index=indices.west_cell,
        params=params,
        vertical_pocket_front_height=0.5 * dz,
        vertical_liquid_surface_height=0.356,
        vertical_branch_confined=True,
        vertical_branch_receiving_hint=True,
        horizontal_downstream_front_position=indices.face_x,
        horizontal_downstream_topology_front_position=indices.face_x,
        prefer_vertical_branch=True,
    )

    horizontal_loss = horizontal_mass_before - float(
        np.sum(result.horizontal_mass)
    )
    vertical_gain = float(np.sum(result.vertical_total_mass)) - vertical_mass_before
    assert result.junction_mouth_area > 0.0
    assert float(np.sum(result.vertical_tracer_mass)) > 0.0
    assert horizontal_loss > 0.0
    assert math.isclose(horizontal_loss, vertical_gain, rel_tol=0.0, abs_tol=2.0e-13)
    assert abs(result.total_mass_error) < 2.0e-13
    assert abs(result.tracer_mass_error) < 2.0e-13
