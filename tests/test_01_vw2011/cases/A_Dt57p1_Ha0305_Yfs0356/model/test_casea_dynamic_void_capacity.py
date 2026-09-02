"""Tests for the mass-backed dynamic riser gas-void capacity prototype."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_dynamic_void_capacity import (  # noqa: E402
    compute_dynamic_material_void_capacity,
)
from casea_vertical_twostream_fv import (  # noqa: E402
    DirectionalBoundaryFlux,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    advance_vertical_two_stream_fv,
    hydrostatic_face_pressures,
)


R_GAS = 287.05
T_GAS = 293.0
P_ATM = 101_325.0


def _gas_mass(void_area: float, dz: float, pressure: float = P_ATM) -> float:
    return pressure * void_area * dz / (R_GAS * T_GAS)


def test_material_gas_capacity_is_the_local_isothermal_eos_volume() -> None:
    full = 6.0e-3
    dz = 0.02
    void = np.array([0.10, 0.25, 0.40]) * full
    mass = P_ATM * void * dz / (R_GAS * T_GAS)
    liquid = np.array([0.50, 0.60, 0.20]) * full

    result = compute_dynamic_material_void_capacity(
        gas_mass=mass,
        tracer_mass=mass,
        liquid_pressure_target=P_ATM,
        current_liquid_area=liquid,
        full_area=full,
        cell_length=dz,
    )

    np.testing.assert_allclose(result.eos_void_area, void, rtol=2.0e-15)
    np.testing.assert_allclose(result.required_void_area, void, rtol=2.0e-15)
    np.testing.assert_allclose(
        result.liquid_capacity_area,
        full - void,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.available_liquid_filling_area,
        full - void - liquid,
        rtol=2.0e-15,
    )


def test_capacity_changes_with_mass_and_pressure_instead_of_freezing_a_corridor() -> None:
    full = 6.0e-3
    dz = 0.02
    base_void = 0.30 * full
    base_mass = _gas_mass(base_void, dz)
    liquid = [0.20 * full]

    base = compute_dynamic_material_void_capacity(
        gas_mass=[base_mass],
        tracer_mass=[base_mass],
        liquid_pressure_target=[P_ATM],
        current_liquid_area=liquid,
        full_area=full,
        cell_length=dz,
    )
    compressed = compute_dynamic_material_void_capacity(
        gas_mass=[base_mass],
        tracer_mass=[base_mass],
        liquid_pressure_target=[2.0 * P_ATM],
        current_liquid_area=liquid,
        full_area=full,
        cell_length=dz,
    )
    doubled_mass = compute_dynamic_material_void_capacity(
        gas_mass=[2.0 * base_mass],
        tracer_mass=[2.0 * base_mass],
        liquid_pressure_target=[P_ATM],
        current_liquid_area=liquid,
        full_area=full,
        cell_length=dz,
    )

    assert compressed.required_void_area[0] == pytest.approx(
        0.5 * base.required_void_area[0]
    )
    assert doubled_mass.required_void_area[0] == pytest.approx(
        2.0 * base.required_void_area[0]
    )
    assert compressed.liquid_capacity_area[0] > base.liquid_capacity_area[0]
    assert doubled_mass.liquid_capacity_area[0] < base.liquid_capacity_area[0]
    assert base.liquid_capacity_area[0] != pytest.approx(0.80 * full)


def test_nonmaterial_positivity_mass_does_not_create_a_fixed_gas_corridor() -> None:
    full = 6.0e-3
    result = compute_dynamic_material_void_capacity(
        gas_mass=[1.0e-5, 1.0e-5],
        tracer_mass=[0.0, 1.0e-8],
        liquid_pressure_target=P_ATM,
        current_liquid_area=[0.40 * full, 0.40 * full],
        full_area=full,
        cell_length=0.02,
        tracer_mass_tolerance=1.0e-10,
    )

    assert not result.material_gas_mask[0]
    assert result.liquid_capacity_area[0] == pytest.approx(full)
    assert result.required_void_area[0] == 0.0
    assert result.material_gas_mask[1]
    assert result.required_void_area[1] > 0.0
    assert result.liquid_capacity_area[1] < full


def test_swept_topology_floor_keeps_only_the_declared_capillary_void() -> None:
    full = 6.0e-3
    floor = np.array([0.0, 0.01 * full, 0.005 * full])
    result = compute_dynamic_material_void_capacity(
        gas_mass=[0.0, 0.0, 0.0],
        tracer_mass=[0.0, 0.0, 0.0],
        liquid_pressure_target=P_ATM,
        current_liquid_area=[0.40 * full] * 3,
        full_area=full,
        cell_length=0.02,
        minimum_topology_void_area=floor,
    )

    np.testing.assert_array_equal(
        result.topology_void_mask,
        [False, True, True],
    )
    np.testing.assert_allclose(result.minimum_topology_void_area, floor)
    np.testing.assert_allclose(result.required_void_area, floor)
    np.testing.assert_allclose(result.liquid_capacity_area, full - floor)
    assert not np.any(result.material_gas_mask)


def test_already_overcompressed_cell_is_not_clipped_or_filled_further() -> None:
    full = 6.0e-3
    dz = 0.02
    required_void = 0.25 * full
    mass = _gas_mass(required_void, dz)
    current_liquid = 0.90 * full

    result = compute_dynamic_material_void_capacity(
        gas_mass=[mass],
        tracer_mass=[mass],
        liquid_pressure_target=P_ATM,
        current_liquid_area=[current_liquid],
        full_area=full,
        cell_length=dz,
    )

    assert result.liquid_capacity_area[0] == pytest.approx(current_liquid)
    assert result.reserved_void_area[0] == pytest.approx(0.10 * full)
    assert result.compression_deficit_area[0] == pytest.approx(0.15 * full)
    assert result.available_liquid_filling_area[0] == 0.0


def test_twostream_capacity_projection_cannot_erase_material_void_in_one_step() -> None:
    diameter = 0.094
    full = 0.25 * math.pi * diameter**2
    dz = 0.10
    params = VerticalTwoStreamParameters(
        cell_count=1,
        cell_length=dz,
        diameter=diameter,
        liquid_density=998.0,
        gravity=9.81,
    )
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.40 * full],
        upward_discharge=[0.0],
        downward_area=[0.35 * full],
        downward_discharge=[0.0],
    )
    material_void = 0.20 * full
    mass = _gas_mass(material_void, dz)
    capacity = compute_dynamic_material_void_capacity(
        gas_mass=[mass],
        tracer_mass=[mass],
        liquid_pressure_target=P_ATM,
        current_liquid_area=state.liquid_area,
        full_area=full,
        cell_length=dz,
    )
    # Request enough bottom inflow to close the complete bore in one step.
    # The shared FV projection may accept only the EOS-compatible 5% fill.
    requested = 10.0 * full
    boundary = DirectionalBoundaryFlux(
        upward_rate=requested,
        upward_speed=requested / state.upward_area[0],
    )
    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=hydrostatic_face_pressures(
            params,
            bottom_pressure=P_ATM + 998.0 * 9.81 * dz,
        ),
        boundaries=VerticalTwoStreamBoundaries(bottom=boundary),
        liquid_capacity_area=capacity.liquid_capacity_area,
    )

    final_void = full - result.state.liquid_area[0]
    assert final_void >= material_void - params.packing_tolerance - 1.0e-15
    assert result.state.liquid_area[0] < full


def test_cell_expansion_beyond_one_bore_is_reported_without_negative_capacity() -> None:
    full = 6.0e-3
    dz = 0.02
    eos_void = 1.40 * full
    mass = _gas_mass(eos_void, dz)

    result = compute_dynamic_material_void_capacity(
        gas_mass=[mass],
        tracer_mass=[mass],
        liquid_pressure_target=P_ATM,
        current_liquid_area=[0.0],
        full_area=full,
        cell_length=dz,
    )

    assert result.required_void_area[0] == pytest.approx(full)
    assert result.liquid_capacity_area[0] == 0.0
    assert result.cell_expansion_excess_area[0] == pytest.approx(0.40 * full)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"tracer_mass": [2.0e-6], "gas_mass": [1.0e-6]}, "exceed"),
        ({"liquid_pressure_target": [0.0]}, "pressure"),
        ({"current_liquid_area": [7.0e-3]}, "outside"),
    ],
)
def test_invalid_conservative_or_thermodynamic_states_are_rejected(
    overrides: dict[str, list[float]],
    message: str,
) -> None:
    arguments = {
        "gas_mass": [1.0e-6],
        "tracer_mass": [1.0e-6],
        "liquid_pressure_target": [P_ATM],
        "current_liquid_area": [3.0e-3],
        "full_area": 6.0e-3,
        "cell_length": 0.02,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        compute_dynamic_material_void_capacity(**arguments)
