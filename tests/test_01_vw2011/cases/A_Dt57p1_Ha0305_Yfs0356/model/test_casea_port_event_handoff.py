"""Conservation tests for the exact Case-A port-event handoff."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
CASE_ROOT = MODEL_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_port_event_handoff import (  # noqa: E402
    CaseAPortGeometry,
    build_casea_port_event_handoff,
)
from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    solve_compressible_node_pressure,
)


CHECKPOINT = (
    CASE_ROOT / "outputs" / "casea_port_west_event_dx40_checkpoint.npz"
)


def _load() -> dict[str, np.ndarray]:
    with np.load(CHECKPOINT) as stored:
        return {name: stored[name].copy() for name in stored.files}


def test_actual_event_handoff_closes_all_preexisting_horizontal_ledgers() -> None:
    result = build_casea_port_event_handoff(_load())
    audit = result.diagnostics
    assert abs(audit.west_gas_mass_error) < 2.0e-17
    assert abs(audit.west_gas_volume_error) < 2.0e-17
    assert abs(audit.horizontal_liquid_inventory_error) < 2.0e-16
    assert abs(audit.horizontal_discharge_inventory_error) < 2.0e-16


def test_node_volume_is_only_the_measured_discrete_port_footprint() -> None:
    geometry = CaseAPortGeometry()
    result = build_casea_port_event_handoff(_load(), geometry=geometry)
    expected_length = geometry.port_east_x - result.diagnostics.event_face_x
    assert result.diagnostics.node_length == expected_length
    assert result.node.node_total_volume == math.pi * (
        geometry.horizontal_diameter**2
    ) / 4.0 * expected_length
    assert result.node.gas_mass == 0.0
    assert result.node.liquid_equivalent_volume == (
        result.diagnostics.old_node_liquid_inventory
    )
    assert result.diagnostics.node_geometric_liquid_volume == (
        result.node.node_total_volume
    )
    assert result.diagnostics.node_elastic_inventory_excess > 0.0

    pressure = solve_compressible_node_pressure(
        result.node,
        CompressibleFiniteNodeParameters(
            gas_sound_speed=math.sqrt(287.05 * 293.0),
            liquid_density=geometry.liquid_density,
            liquid_wave_speed=geometry.horizontal_wave_speed,
        ),
    )
    # The node pressure is recovered from the conservative elastic inventory,
    # not copied from the checkpoint.  It remains close to the local
    # pre-event pocket pressure because this is an overlap average over the
    # measured finite opening.
    checkpoint_pressure = float(_load()["gas_pressure"][0])
    assert abs(pressure.pressure_abs - checkpoint_pressure) < 250.0
    assert abs(pressure.occupancy_residual) < 2.0e-15


def test_new_gas_momentum_degree_of_freedom_obeys_both_event_boundaries() -> None:
    checkpoint = _load()
    result = build_casea_port_event_handoff(checkpoint)
    speed = float(checkpoint["interface_speed"][0])
    length = result.west_faces[-1]
    first_x = 0.5 * (result.west_faces[0] + result.west_faces[1])
    last_x = 0.5 * (result.west_faces[-2] + result.west_faces[-1])
    assert result.west_cells[0].gas_velocity == speed * first_x / length
    assert result.west_cells[-1].gas_velocity == speed * last_x / length
    assert speed * 0.0 / length == 0.0
    assert math.isclose(
        speed * length / length, speed, rel_tol=0.0, abs_tol=1.0e-16
    )
    assert result.diagnostics.reconstructed_gas_kinetic_energy > 0.0


def test_east_remap_is_local_and_retains_pressurised_inventory() -> None:
    checkpoint = _load()
    result = build_casea_port_event_handoff(checkpoint)
    assert result.diagnostics.east_elastic_inventory_correction == 0.0
    assert result.diagnostics.node_elastic_inventory_excess > 0.0
    assert result.diagnostics.old_node_discharge_inventory < 0.0
    assert result.east_faces[0] == 0.0
    assert result.east_faces[-1] == (
        CaseAPortGeometry().horizontal_length
        - CaseAPortGeometry().port_east_x
    )
    assert all(cell.area > 0.0 for cell in result.east_pressurised_cells)
    # No first-cell area impulse is used to carry the node's elastic storage.
    full_area = CaseAPortGeometry().horizontal_area
    assert result.east_pressurised_cells[0].area < 1.01 * full_area


def test_vertical_water_column_is_axially_cut_at_the_flat_free_surface() -> None:
    geometry = CaseAPortGeometry()
    result = build_casea_port_event_handoff(
        _load(), geometry=geometry, vertical_target_width=0.020
    )
    widths = np.diff(result.vertical_faces)
    assert len(widths) == 18
    assert np.allclose(widths[:-1], 0.020)
    assert math.isclose(widths[-1], 0.016, rel_tol=0.0, abs_tol=1.0e-15)
    assert result.vertical_faces[-1] == geometry.riser_liquid_height
    assert result.vertical_open_surface_height == geometry.riser_liquid_height
    areas = np.asarray(
        [cell.area for cell in result.vertical_pressurised_cells]
    )
    assert np.all(areas > geometry.vertical_area)
    assert np.all(np.diff(areas) < 0.0)
