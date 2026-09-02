from __future__ import annotations

import inspect
import math

import pytest

from casea_vertical_twostream_closures import (
    GasVoidStateError,
    IsothermalGasClosureParameters,
    adapt_gas_void_and_pressure_faces,
    advance_post_event_core_film_stage,
    advance_taylor_sweep_geometry,
    atmospheric_top_liquid_outflow,
    coaxial_core_film_geometry,
    displace_taylor_core_liquid_into_unswept_cells,
    extend_taylor_sweep_in_persistent_state,
)
from casea_vertical_twostream_fv import (
    DirectionalBoundaryFlux,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    implicit_physical_three_body_drag_exchange,
)


def _parameters(*, cell_count: int = 4) -> VerticalTwoStreamParameters:
    return VerticalTwoStreamParameters(
        cell_count=cell_count,
        cell_length=0.05,
        diameter=0.0571,
        gravity=9.81,
        wall_friction_up=0.0,
        wall_friction_down=0.0,
        interstream_drag=0.0,
    )


def _gas_inventory_for_cell_pressures(
    pressures: tuple[float, ...],
    gas_areas: tuple[float, ...],
    parameters: VerticalTwoStreamParameters,
    gas: IsothermalGasClosureParameters,
) -> tuple[float, ...]:
    return tuple(
        pressure * area * parameters.cell_length / gas.sound_speed_squared
        for pressure, area in zip(pressures, gas_areas)
    )


def test_coaxial_geometry_recovers_exact_annular_areas_and_perimeters() -> None:
    params = _parameters(cell_count=1)
    radius_core = 0.010
    radius_film_inner = 0.025
    area_up = math.pi * radius_core**2
    area_down = params.full_area - math.pi * radius_film_inner**2
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[area_up],
        upward_discharge=[0.0],
        downward_area=[area_down],
        downward_discharge=[0.0],
    )

    geometry = coaxial_core_film_geometry(state, params)

    expected_gas = math.pi * (radius_film_inner**2 - radius_core**2)
    expected_p_up = 2.0 * math.pi * radius_core
    expected_p_down = 2.0 * math.pi * radius_film_inner
    assert geometry.gas_area[0] == pytest.approx(expected_gas, abs=1.0e-15)
    assert geometry.upward_core_radius[0] == pytest.approx(radius_core)
    assert geometry.film_inner_radius[0] == pytest.approx(radius_film_inner)
    assert geometry.upward_gas_interface_perimeter[0] == pytest.approx(expected_p_up)
    assert geometry.downward_gas_interface_perimeter[0] == pytest.approx(expected_p_down)
    assert geometry.gas_hydraulic_diameter[0] == pytest.approx(
        4.0 * expected_gas / (expected_p_up + expected_p_down)
    )
    assert geometry.downward_wall_perimeter[0] == pytest.approx(
        math.pi * params.diameter
    )


def test_coaxial_geometry_accepts_roundoff_scale_closed_gas_gap() -> None:
    """Packing and coaxial-radius checks must share one tolerance."""

    params = VerticalTwoStreamParameters(
        cell_count=1,
        cell_length=0.01,
        diameter=0.0571,
        packing_tolerance=2.0e-12,
    )
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.60 * full],
        upward_discharge=[1.0e-5],
        downward_area=[0.40 * full + 1.0e-12],
        downward_discharge=[-1.0e-5],
    )

    geometry = coaxial_core_film_geometry(state, params)

    assert geometry.gas_area[0] == 0.0
    assert geometry.film_inner_radius[0] == pytest.approx(
        geometry.upward_core_radius[0], rel=0.0, abs=1.0e-14
    )


def test_taylor_sweep_extends_only_new_cells_and_preserves_local_totals() -> None:
    params = _parameters(cell_count=3)
    area = 0.70 * params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.50 * area, area, area],
        upward_discharge=[2.0e-5, 0.0, 0.0],
        downward_area=[0.20 * area, 0.0, 0.0],
        downward_discharge=[-1.0e-5, 0.0, 0.0],
    )
    old_totals = state.liquid_area
    old_momenta = state.liquid_discharge

    first = extend_taylor_sweep_in_persistent_state(
        state,
        params,
        previous_swept_fraction=[0.5, 0.0, 0.0],
        new_swept_fraction=[0.5, 0.4, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )

    # Cell zero was swept in an earlier stage and is copied byte-for-byte.
    assert first.state.upward_area[0] == state.upward_area[0]
    assert first.state.upward_discharge[0] == state.upward_discharge[0]
    assert first.state.downward_area[0] == state.downward_area[0]
    assert first.state.downward_discharge[0] == state.downward_discharge[0]
    # Only the newly swept second cell acquires a falling-film inventory.
    assert first.added_film_area[1] > 0.0
    assert first.added_film_discharge[1] < 0.0
    assert first.added_film_area[2] == 0.0
    assert first.state.liquid_area == pytest.approx(old_totals, abs=1.0e-15)
    assert first.state.liquid_discharge == pytest.approx(old_momenta, abs=1.0e-15)
    assert first.area_residual == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-15)
    assert first.momentum_residual == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-15)

    # Repeating the same sweep event is exactly idempotent.
    repeated = extend_taylor_sweep_in_persistent_state(
        first.state,
        params,
        previous_swept_fraction=first.new_swept_fraction,
        new_swept_fraction=first.new_swept_fraction,
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    assert repeated.state == first.state


def test_taylor_sweep_collapses_orphaned_momentum_when_upward_label_drains() -> None:
    params = _parameters(cell_count=1)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-5],
        upward_discharge=[5.0e-7],
        downward_area=[1.0e-3],
        downward_discharge=[-2.0e-3],
    )
    old_area = state.liquid_area[0]
    old_discharge = state.liquid_discharge[0]

    result = extend_taylor_sweep_in_persistent_state(
        state,
        params,
        previous_swept_fraction=[0.0],
        new_swept_fraction=[1.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )

    assert result.state.upward_area == (0.0,)
    assert result.state.upward_discharge == (0.0,)
    assert result.state.downward_area[0] == pytest.approx(old_area, abs=1.0e-15)
    assert result.state.downward_discharge[0] == pytest.approx(
        old_discharge, abs=1.0e-15
    )
    assert result.area_residual == pytest.approx((0.0,), abs=1.0e-15)
    assert result.momentum_residual == pytest.approx((0.0,), abs=1.0e-15)


def test_partial_first_cell_credits_existing_gas_void_without_repeated_displacement() -> None:
    params = _parameters(cell_count=3)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.8 * full] * 3,
        upward_discharge=[0.8 * full * 0.20] * 3,
        downward_area=[0.0] * 3,
        downward_discharge=[0.0] * 3,
    )
    old_volume = params.cell_length * sum(state.liquid_area)
    old_momentum = params.cell_length * sum(state.liquid_discharge)

    result = advance_taylor_sweep_geometry(
        state,
        params,
        previous_swept_fraction=[0.0, 0.0, 0.0],
        new_swept_fraction=[0.25, 0.0, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    displacement = result.gas_core_displacement

    # The source already contains exactly the 0.20 A gas area required by
    # S_new=0.25 and alpha_core=0.8, so the sweep must not remove it again.
    assert displacement.requested_core_area == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-15
    )
    assert displacement.opened_core_area == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-15
    )
    assert displacement.received_upward_area == pytest.approx(
        (0.0, 0.0, 0.0), abs=1.0e-15
    )
    assert result.state.downward_area == result.film_extension.state.downward_area
    assert result.state.downward_discharge == result.film_extension.state.downward_discharge
    assert params.cell_length * sum(result.state.liquid_area) == pytest.approx(
        old_volume, abs=1.0e-15
    )
    assert params.cell_length * sum(result.state.liquid_discharge) == pytest.approx(
        old_momentum, abs=1.0e-15
    )
    assert displacement.overflow_liquid_volume == 0.0
    assert displacement.volume_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert displacement.momentum_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_partial_first_cell_without_existing_void_keeps_incremental_displacement() -> None:
    params = _parameters(cell_count=3)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[full, 0.8 * full, 0.8 * full],
        upward_discharge=[full * 0.20, 0.8 * full * 0.20, 0.8 * full * 0.20],
        downward_area=[0.0] * 3,
        downward_discharge=[0.0] * 3,
    )
    old_volume = params.cell_length * sum(state.liquid_area)
    old_momentum = params.cell_length * sum(state.liquid_discharge)

    result = advance_taylor_sweep_geometry(
        state,
        params,
        previous_swept_fraction=[0.0, 0.0, 0.0],
        new_swept_fraction=[0.25, 0.0, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    displacement = result.gas_core_displacement

    assert displacement.requested_core_area == pytest.approx(
        (0.20 * full, 0.0, 0.0), abs=1.0e-15
    )
    assert displacement.opened_core_area == pytest.approx(
        displacement.requested_core_area, abs=1.0e-15
    )
    assert displacement.received_upward_area == pytest.approx(
        (0.0, 0.20 * full, 0.0), abs=1.0e-15
    )
    assert displacement.source_shortfall_volume == 0.0
    assert displacement.overflow_liquid_volume == 0.0
    assert params.cell_length * sum(result.state.liquid_area) == pytest.approx(
        old_volume, abs=1.0e-15
    )
    assert params.cell_length * sum(result.state.liquid_discharge) == pytest.approx(
        old_momentum, abs=1.0e-15
    )
    assert displacement.volume_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert displacement.momentum_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_multiple_newly_swept_cells_fill_nearest_unswept_receivers() -> None:
    params = _parameters(cell_count=4)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[full, full, 0.30 * full, 0.30 * full],
        upward_discharge=[0.10 * full, 0.20 * full, 0.0, 0.0],
        downward_area=[0.0] * 4,
        downward_discharge=[0.0] * 4,
    )

    result = advance_taylor_sweep_geometry(
        state,
        params,
        previous_swept_fraction=[0.0, 0.0, 0.0, 0.0],
        new_swept_fraction=[1.0, 0.5, 0.0, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    displacement = result.gas_core_displacement

    assert displacement.opened_core_area == pytest.approx(
        (0.80 * full, 0.40 * full, 0.0, 0.0), abs=1.0e-15
    )
    # Receiver 2 is filled first (0.70 A capacity), then receiver 3.
    assert displacement.received_upward_area == pytest.approx(
        (0.0, 0.0, 0.70 * full, 0.50 * full), abs=1.0e-15
    )
    assert displacement.overflow_liquid_volume == 0.0
    assert displacement.source_shortfall_volume == 0.0
    assert displacement.volume_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert displacement.momentum_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_full_downward_activation_leaves_film_and_opens_gas_core() -> None:
    params = _parameters(cell_count=2)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.0, 0.10 * full],
        upward_discharge=[0.0, 0.0],
        downward_area=[full, 0.0],
        downward_discharge=[-0.20 * full, 0.0],
    )
    initial_volume = params.cell_length * sum(state.liquid_area)
    initial_momentum = params.cell_length * sum(state.liquid_discharge)

    result = advance_taylor_sweep_geometry(
        state,
        params,
        previous_swept_fraction=[0.0, 0.0],
        new_swept_fraction=[1.0, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    displacement = result.gas_core_displacement

    assert displacement.removed_upward_area[0] == 0.0
    assert displacement.protected_film_area[0] == pytest.approx(0.20 * full)
    assert displacement.film_requirement_shortfall_area[0] == 0.0
    assert displacement.removed_downward_area[0] == pytest.approx(0.80 * full)
    assert displacement.removed_downward_discharge[0] == pytest.approx(
        -0.16 * full
    )
    assert result.state.downward_area[0] == pytest.approx(0.20 * full)
    assert result.state.downward_discharge[0] == pytest.approx(-0.04 * full)
    assert result.state.upward_area[0] == 0.0
    assert displacement.received_downward_area[1] == pytest.approx(0.80 * full)
    assert displacement.received_downward_discharge[1] == pytest.approx(
        -0.16 * full
    )
    assert displacement.opened_core_area[0] == pytest.approx(0.80 * full)
    assert displacement.source_shortfall_volume == 0.0
    assert displacement.overflow_liquid_volume == 0.0
    assert params.cell_length * sum(result.state.liquid_area) == pytest.approx(
        initial_volume, abs=1.0e-15
    )
    assert params.cell_length * sum(result.state.liquid_discharge) == pytest.approx(
        initial_momentum, abs=1.0e-15
    )
    assert displacement.volume_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert displacement.momentum_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_gas_core_displacement_is_idempotent_for_unchanged_sweep() -> None:
    params = _parameters(cell_count=3)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.5 * full, 0.4 * full, 0.3 * full],
        upward_discharge=[1.0e-5, 2.0e-5, 3.0e-5],
        downward_area=[0.2 * full, 0.1 * full, 0.0],
        downward_discharge=[-1.0e-5, -5.0e-6, 0.0],
    )

    result = displace_taylor_core_liquid_into_unswept_cells(
        state,
        params,
        previous_swept_fraction=[1.0, 0.4, 0.0],
        new_swept_fraction=[1.0, 0.4, 0.0],
        taylor_core_area_fraction=0.8,
    )

    assert result.state == state
    assert result.opened_core_volume == 0.0
    assert result.deposited_liquid_volume == 0.0
    assert result.overflow_liquid_volume == 0.0
    assert result.volume_residual_including_overflow == 0.0
    assert result.momentum_residual_including_overflow == 0.0


def test_receiver_shortage_returns_explicit_overflow_without_loss() -> None:
    params = _parameters(cell_count=2)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[full, 0.95 * full],
        upward_discharge=[0.20 * full, 0.0],
        downward_area=[0.0, 0.0],
        downward_discharge=[0.0, 0.0],
    )

    result = advance_taylor_sweep_geometry(
        state,
        params,
        previous_swept_fraction=[0.0, 0.0],
        new_swept_fraction=[0.5, 0.0],
        taylor_core_area_fraction=0.8,
        taylor_rise_velocity=0.25,
    )
    displacement = result.gas_core_displacement
    expected_opened = 0.40 * full
    expected_deposit = 0.05 * full
    expected_overflow = (expected_opened - expected_deposit) * params.cell_length

    assert displacement.opened_core_area[0] == pytest.approx(expected_opened)
    assert displacement.received_upward_area[1] == pytest.approx(expected_deposit)
    assert displacement.overflow_liquid_volume == pytest.approx(expected_overflow)
    assert displacement.overflow_kinematic_momentum > 0.0
    assert displacement.has_overflow
    assert displacement.volume_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert displacement.momentum_residual_including_overflow == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert result.state.downward_area == result.film_extension.state.downward_area
    assert result.state.downward_discharge == result.film_extension.state.downward_discharge


def test_gas_adapter_uses_inventory_eos_and_atmospheric_top_face() -> None:
    params = _parameters(cell_count=3)
    gas = IsothermalGasClosureParameters()
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.10 * params.full_area] * 3,
        upward_discharge=[0.0] * 3,
        downward_area=[0.20 * params.full_area] * 3,
        downward_discharge=[0.0] * 3,
    )
    geometry = coaxial_core_film_geometry(state, params)
    cell_pressure = (101900.0, 101600.0, 101400.0)
    mass = _gas_inventory_for_cell_pressures(cell_pressure, geometry.gas_area, params, gas)

    adapter = adapt_gas_void_and_pressure_faces(
        state,
        params,
        gas_mass=mass,
        gas_momentum=[0.0] * 3,
        gas=gas,
        bottom_pressure=102100.0,
    )

    assert adapter.gas_pressure_cells == pytest.approx(cell_pressure)
    assert adapter.common_pressure_faces == pytest.approx(
        (102100.0, 101750.0, 101500.0, gas.atmospheric_pressure)
    )
    assert adapter.physical_drag_state.gas_area == pytest.approx(geometry.gas_area)
    assert adapter.physical_drag_state.upward_interface_perimeter == pytest.approx(
        geometry.upward_gas_interface_perimeter
    )
    assert adapter.physical_drag_state.upward_hydraulic_diameter == pytest.approx(
        geometry.gas_hydraulic_diameter
    )


def test_gas_adapter_rejects_massless_void_instead_of_injecting_air() -> None:
    params = _parameters(cell_count=1)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.1 * params.full_area],
        upward_discharge=[0.0],
        downward_area=[0.1 * params.full_area],
        downward_discharge=[0.0],
    )
    with pytest.raises(GasVoidStateError, match="no gas mass"):
        adapt_gas_void_and_pressure_faces(
            state,
            params,
            gas_mass=[0.0],
            gas_momentum=[0.0],
        )


def test_adapter_geometry_closes_nonzero_three_body_drag_momentum() -> None:
    params = _parameters(cell_count=1)
    gas = IsothermalGasClosureParameters()
    area_up = 0.15 * params.full_area
    area_down = 0.20 * params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[area_up],
        upward_discharge=[area_up * 0.30],
        downward_area=[area_down],
        downward_discharge=[-area_down * 0.20],
    )
    geometry = coaxial_core_film_geometry(state, params)
    cell_pressure = (gas.atmospheric_pressure,)
    mass = _gas_inventory_for_cell_pressures(cell_pressure, geometry.gas_area, params, gas)
    gas_velocity = 0.50
    gas_momentum = (mass[0] * gas_velocity,)
    adapter = adapt_gas_void_and_pressure_faces(
        state,
        params,
        gas_mass=mass,
        gas_momentum=gas_momentum,
        gas=gas,
    )
    initial_total = (
        gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * (state.upward_discharge[0] + state.downward_discharge[0])
    )

    result = implicit_physical_three_body_drag_exchange(
        state,
        params,
        adapter.physical_drag_state,
        dt=2.0e-3,
    )
    final_total = (
        result.gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * (result.state.upward_discharge[0] + result.state.downward_discharge[0])
    )

    assert adapter.geometry.upward_gas_interface_perimeter[0] > 0.0
    assert adapter.geometry.downward_gas_interface_perimeter[0] > 0.0
    assert adapter.geometry.gas_hydraulic_diameter[0] > 0.0
    assert final_total == pytest.approx(initial_total, abs=2.0e-14)
    assert result.total_momentum_residual == pytest.approx(0.0, abs=2.0e-14)


def test_atmospheric_top_boundary_is_liquid_outflow_only() -> None:
    params = _parameters(cell_count=2)
    area_up = 0.15 * params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[area_up, area_up],
        upward_discharge=[0.0, 2.0e-5],
        downward_area=[0.10 * params.full_area, 0.10 * params.full_area],
        downward_discharge=[-1.0e-5, -1.0e-5],
    )

    boundary = atmospheric_top_liquid_outflow(state, params)

    assert boundary.flux.upward_rate == pytest.approx(2.0e-5)
    assert boundary.flux.upward_speed == pytest.approx(2.0e-5 / area_up)
    assert boundary.flux.downward_rate == 0.0
    assert boundary.flux.net_rate == pytest.approx(2.0e-5)
    assert boundary.atmospheric_pressure == pytest.approx(101325.0)


def test_post_event_stage_preserves_stationary_hydrostatic_partition() -> None:
    params = _parameters(cell_count=4)
    gas = IsothermalGasClosureParameters()
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.12 * params.full_area] * 4,
        upward_discharge=[0.0] * 4,
        downward_area=[0.18 * params.full_area] * 4,
        downward_discharge=[0.0] * 4,
    )
    geometry = coaxial_core_film_geometry(state, params)
    height = params.cell_count * params.cell_length
    pressure_faces = tuple(
        gas.atmospheric_pressure
        + params.liquid_density * params.gravity * (height - face * params.cell_length)
        for face in range(params.cell_count + 1)
    )
    cell_pressure = tuple(
        0.5 * (pressure_faces[cell] + pressure_faces[cell + 1])
        for cell in range(params.cell_count)
    )
    mass = _gas_inventory_for_cell_pressures(cell_pressure, geometry.gas_area, params, gas)

    result = advance_post_event_core_film_stage(
        state,
        params,
        dt=1.0e-3,
        gas_mass=mass,
        gas_momentum=[0.0] * 4,
        gas=gas,
        bottom_pressure=pressure_faces[0],
        apply_physical_drag=True,
    )

    assert result.pressure_before.common_pressure_faces == pytest.approx(
        pressure_faces, abs=1.0e-10
    )
    assert result.state.upward_area == pytest.approx(state.upward_area, abs=1.0e-15)
    assert result.state.downward_area == pytest.approx(state.downward_area, abs=1.0e-15)
    assert result.state.upward_discharge == pytest.approx((0.0,) * 4, abs=1.0e-15)
    assert result.state.downward_discharge == pytest.approx((0.0,) * 4, abs=1.0e-15)
    assert result.gas_momentum == pytest.approx((0.0,) * 4, abs=1.0e-15)
    assert result.inventory.total_volume_residual == pytest.approx(0.0, abs=1.0e-15)


def test_two_prognostic_inventories_counterflow_without_area_setpoint() -> None:
    params = _parameters(cell_count=2)
    gas = IsothermalGasClosureParameters()
    up_area = 0.12 * params.full_area
    down_area = 0.10 * params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[up_area, up_area],
        upward_discharge=[1.0e-5, 0.0],
        downward_area=[down_area, down_area],
        downward_discharge=[0.0, -8.0e-6],
    )
    geometry = coaxial_core_film_geometry(state, params)
    height = params.cell_count * params.cell_length
    pressure_faces = tuple(
        gas.atmospheric_pressure
        + params.liquid_density * params.gravity * (height - face * params.cell_length)
        for face in range(params.cell_count + 1)
    )
    cell_pressure = tuple(
        0.5 * (pressure_faces[cell] + pressure_faces[cell + 1])
        for cell in range(params.cell_count)
    )
    mass = _gas_inventory_for_cell_pressures(cell_pressure, geometry.gas_area, params, gas)
    initial_up_volume = params.cell_length * sum(state.upward_area)
    initial_down_volume = params.cell_length * sum(state.downward_area)

    result = advance_post_event_core_film_stage(
        state,
        params,
        dt=1.0e-3,
        gas_mass=mass,
        gas_momentum=[0.0, 0.0],
        gas=gas,
        bottom_pressure=pressure_faces[0],
        apply_physical_drag=False,
    )

    assert params.cell_length * sum(result.state.upward_area) == pytest.approx(
        initial_up_volume, abs=1.0e-15
    )
    assert params.cell_length * sum(result.state.downward_area) == pytest.approx(
        initial_down_volume, abs=1.0e-15
    )
    assert result.state.upward_area[0] < state.upward_area[0]
    assert result.state.upward_area[1] > state.upward_area[1]
    assert result.state.downward_area[0] > state.downward_area[0]
    assert result.state.downward_area[1] < state.downward_area[1]
    assert result.inventory.total_volume_residual == pytest.approx(0.0, abs=1.0e-15)


def test_top_liquid_exit_is_the_only_domain_volume_change() -> None:
    params = _parameters(cell_count=2)
    gas = IsothermalGasClosureParameters()
    up_area = 0.10 * params.full_area
    up_rate = 1.0e-5
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[up_area, up_area],
        upward_discharge=[up_rate, up_rate],
        downward_area=[0.0, 0.0],
        downward_discharge=[0.0, 0.0],
    )
    geometry = coaxial_core_film_geometry(state, params)
    height = params.cell_count * params.cell_length
    pressure_faces = tuple(
        gas.atmospheric_pressure
        + params.liquid_density * params.gravity * (height - face * params.cell_length)
        for face in range(params.cell_count + 1)
    )
    cell_pressure = tuple(
        0.5 * (pressure_faces[cell] + pressure_faces[cell + 1])
        for cell in range(params.cell_count)
    )
    mass = _gas_inventory_for_cell_pressures(cell_pressure, geometry.gas_area, params, gas)
    dt = 1.0e-3
    initial_volume = params.cell_length * sum(state.liquid_area)

    result = advance_post_event_core_film_stage(
        state,
        params,
        dt=dt,
        gas_mass=mass,
        gas_momentum=[0.0, 0.0],
        gas=gas,
        bottom_pressure=pressure_faces[0],
        apply_physical_drag=False,
    )

    final_volume = params.cell_length * sum(result.state.liquid_area)
    assert final_volume == pytest.approx(initial_volume - dt * up_rate, abs=1.0e-15)
    assert result.top_boundary.flux.downward_rate == 0.0
    assert result.inventory.total_volume_residual == pytest.approx(0.0, abs=1.0e-15)


def test_stage_api_contains_no_result_feedback_controls() -> None:
    forbidden = {
        "event_time",
        "requested_height",
        "reference_field",
        "openfoam_input",
        "display_shape",
    }
    for callable_api in (
        advance_post_event_core_film_stage,
        advance_taylor_sweep_geometry,
        displace_taylor_core_liquid_into_unswept_cells,
    ):
        assert forbidden.isdisjoint(inspect.signature(callable_api).parameters)
