"""Conservation and limiting tests for the isolated riser two-stream core."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_vertical_twostream_fv import (  # noqa: E402
    COMPLETE_CASEA_RISER_READY,
    MISSING_PHYSICAL_CLOSURES,
    TWOSTREAM_FV_CORE_READY,
    DirectionalBoundaryFlux,
    GasMomentumCoupling,
    PhysicalGasInterphaseState,
    VerticalTwoStreamBoundaries,
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    advance_vertical_two_stream_fv,
    bottom_net_receiving_rate_capacity,
    conservative_directional_topology_transfer,
    hydrostatic_face_pressures,
    implicit_physical_three_body_drag_exchange,
    map_taylor_breakthrough_to_twostream,
)


def _parameters(
    *,
    cells: int = 4,
    dz: float = 0.10,
    interstream_drag: float = 0.0,
) -> VerticalTwoStreamParameters:
    return VerticalTwoStreamParameters(
        cell_count=cells,
        cell_length=dz,
        diameter=0.094,
        liquid_density=998.0,
        gravity=9.81,
        wall_friction_up=0.0,
        wall_friction_down=0.0,
        interstream_drag=interstream_drag,
    )


def _hydrostatic(params: VerticalTwoStreamParameters) -> tuple[float, ...]:
    return hydrostatic_face_pressures(params, bottom_pressure=130_000.0)


def test_stationary_hydrostatic_two_stream_state_is_preserved_exactly() -> None:
    params = _parameters(cells=5, dz=0.08)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[2.0e-3] * 5,
        upward_discharge=[0.0] * 5,
        downward_area=[1.1e-3] * 5,
        downward_discharge=[0.0] * 5,
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.02,
        pressure_faces=_hydrostatic(params),
    )

    assert result.state == state
    assert result.ledger.total_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.pressure_gravity_impulse == pytest.approx(0.0, abs=2.0e-15)
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_two_direction_boundary_fluxes_close_each_mass_ledger() -> None:
    params = _parameters(cells=3, dz=0.20)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[2.2e-3] * 3,
        upward_discharge=[1.0e-4] * 3,
        downward_area=[1.0e-3] * 3,
        downward_discharge=[-6.0e-5] * 3,
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=1.0e-4 / 2.2e-3,
            downward_rate=4.0e-5,
            downward_speed=4.0e-5 / 1.0e-3,
        ),
        top=DirectionalBoundaryFlux(
            upward_rate=6.0e-5,
            upward_speed=6.0e-5 / 2.2e-3,
            downward_rate=8.0e-5,
            downward_speed=8.0e-5 / 1.0e-3,
        ),
    )
    dt = 0.05

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert boundaries.bottom.net_rate == pytest.approx(6.0e-5)
    assert boundaries.top.net_rate == pytest.approx(-2.0e-5)
    assert result.ledger.upward_boundary_volume_change == pytest.approx(
        dt * (1.0e-4 - 6.0e-5)
    )
    assert result.ledger.downward_boundary_volume_change == pytest.approx(
        dt * (-4.0e-5 + 8.0e-5)
    )
    assert result.ledger.upward_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.downward_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.total_volume_residual == pytest.approx(0.0, abs=3.0e-18)
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_interstream_exchange_is_equal_opposite_and_reduces_slip() -> None:
    params = _parameters(cells=1, dz=0.25, interstream_drag=5.0)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[2.0e-3],
        upward_discharge=[6.0e-4],
        downward_area=[1.0e-3],
        downward_discharge=[-2.0e-4],
    )
    before_slip = 6.0e-4 / 2.0e-3 - (-2.0e-4 / 1.0e-3)

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
    )
    after_slip = (
        result.state.upward_discharge[0] / result.state.upward_area[0]
        - result.state.downward_discharge[0] / result.state.downward_area[0]
    )

    assert after_slip < before_slip
    assert result.ledger.interstream_upward_impulse < 0.0
    assert result.ledger.interstream_downward_impulse > 0.0
    assert result.ledger.interstream_momentum_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_wall_friction_reduces_both_speed_magnitudes_without_changing_area() -> None:
    params = VerticalTwoStreamParameters(
        cell_count=1,
        cell_length=0.25,
        diameter=0.094,
        wall_friction_up=0.03,
        wall_friction_down=0.04,
    )
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[2.0e-3],
        upward_discharge=[6.0e-4],
        downward_area=[1.0e-3],
        downward_discharge=[-2.0e-4],
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
    )

    assert result.state.upward_area == state.upward_area
    assert result.state.downward_area == state.downward_area
    assert result.state.upward_discharge[0] < state.upward_discharge[0]
    assert abs(result.state.downward_discharge[0]) < abs(state.downward_discharge[0])
    assert result.ledger.wall_impulse < 0.0
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_liquid_gas_drag_returns_the_exact_opposite_gas_impulse() -> None:
    params = _parameters(cells=1, dz=0.30)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3],
        upward_discharge=[2.0e-4],
        downward_area=[1.0e-3],
        downward_discharge=[-2.0e-4],
    )
    gas = GasMomentumCoupling.from_iterables(
        gas_area=[params.full_area - 2.0e-3],
        gas_velocity=[0.05],
        drag_coefficient=1.5,
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
        gas_coupling=gas,
    )

    assert result.ledger.gas_on_liquid_impulse != 0.0
    assert result.ledger.gas_reaction_impulse == pytest.approx(
        -result.ledger.gas_on_liquid_impulse, abs=2.0e-18
    )
    assert result.ledger.liquid_gas_exchange_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_physical_three_body_drag_conserves_gas_and_both_liquid_momenta() -> None:
    params = _parameters(cells=1, dz=0.10)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3],
        upward_discharge=[4.0e-4],
        downward_area=[8.0e-4],
        downward_discharge=[-1.6e-4],
    )
    gas_area = params.full_area - 1.8e-3
    gas_mass = 2.5 * gas_area * params.cell_length
    gas = PhysicalGasInterphaseState.from_iterables(
        gas_mass=[gas_mass],
        gas_momentum=[gas_mass * 1.2],
        gas_area=[gas_area],
        upward_interface_perimeter=[0.045],
        downward_interface_perimeter=[0.15],
        upward_hydraulic_diameter=[0.035],
        downward_hydraulic_diameter=[0.004],
    )
    initial_total = (
        gas.gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * (state.upward_discharge[0] + state.downward_discharge[0])
    )

    result = implicit_physical_three_body_drag_exchange(
        state,
        params,
        gas,
        dt=1.0e-3,
    )
    final_total = (
        result.gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * (
            result.state.upward_discharge[0]
            + result.state.downward_discharge[0]
        )
    )

    assert result.upward_friction_factor[0] > 0.0
    assert result.downward_friction_factor[0] > 0.0
    assert final_total == pytest.approx(initial_total, abs=2.0e-16)
    assert result.cell_momentum_residual[0] == pytest.approx(0.0, abs=2.0e-16)
    assert (
        result.gas_impulse[0]
        + result.upward_liquid_impulse[0]
        + result.downward_liquid_impulse[0]
    ) == pytest.approx(0.0, abs=2.0e-16)


def test_physical_drag_cannot_erase_a_preserved_mouth_corridor() -> None:
    params = _parameters(cells=1, dz=0.10)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3],
        upward_discharge=[1.0e-4],
        downward_area=[8.0e-4],
        downward_discharge=[-1.0e-8],
    )
    gas_area = params.full_area - sum(state.liquid_area)
    gas_mass = 2.0 * gas_area * params.cell_length
    gas = PhysicalGasInterphaseState.from_iterables(
        gas_mass=[gas_mass],
        gas_momentum=[gas_mass * 8.0],
        gas_area=[gas_area],
        upward_interface_perimeter=[0.10],
        downward_interface_perimeter=[0.20],
        upward_hydraulic_diameter=[0.02],
        downward_hydraulic_diameter=[0.004],
    )
    initial_total = (
        gas.gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * state.liquid_discharge[0]
    )
    initial_energy = (
        0.5 * gas.gas_momentum[0] ** 2 / gas.gas_mass[0]
        + 0.5
        * params.liquid_density
        * params.cell_length
        * (
            state.upward_discharge[0] ** 2 / state.upward_area[0]
            + state.downward_discharge[0] ** 2 / state.downward_area[0]
        )
    )

    result = implicit_physical_three_body_drag_exchange(
        state,
        params,
        gas,
        dt=0.05,
        preserve_stopped_partition=[True],
    )
    final_total = (
        result.gas_momentum[0]
        + params.liquid_density
        * params.cell_length
        * result.state.liquid_discharge[0]
    )

    final_energy = (
        0.5 * result.gas_momentum[0] ** 2 / gas.gas_mass[0]
        + 0.5
        * params.liquid_density
        * params.cell_length
        * (
            result.state.upward_discharge[0] ** 2
            / result.state.upward_area[0]
            + (
                result.state.downward_discharge[0] ** 2
                / result.state.downward_area[0]
                if result.state.downward_area[0] > 0.0
                else 0.0
            )
        )
    )

    assert result.state.downward_area[0] > 0.0
    assert result.state.downward_discharge[0] <= 0.0
    assert sum(result.state.liquid_area) == pytest.approx(
        sum(state.liquid_area), abs=2.0e-18
    )
    assert final_total == pytest.approx(initial_total, abs=2.0e-16)
    assert final_energy <= initial_energy + 2.0e-16


def test_preserved_one_sided_reversal_keeps_entropy_stable_residual_area() -> None:
    upward_area = 1.0e-3
    downward_area = 8.0e-4
    upward_discharge = 1.0e-4
    reversed_downward_discharge = 1.0e-6
    energy_before = 0.5 * (
        upward_discharge**2 / upward_area
        + reversed_downward_discharge**2 / downward_area
    )

    result = conservative_directional_topology_transfer(
        upward_area=[upward_area],
        upward_discharge=[upward_discharge],
        downward_area=[downward_area],
        downward_discharge=[reversed_downward_discharge],
        preserve_stopped_partition=[True],
    )
    energy_after = 0.5 * (
        result.state.upward_discharge[0] ** 2
        / result.state.upward_area[0]
    )

    assert 0.0 < result.state.downward_area[0] < downward_area
    assert result.state.downward_discharge == (0.0,)
    assert result.state.liquid_area == pytest.approx([upward_area + downward_area])
    assert result.state.liquid_discharge == pytest.approx(
        [upward_discharge + reversed_downward_discharge]
    )
    assert energy_after <= energy_before + 2.0e-18


def test_preserved_reversal_with_dry_receiver_transfers_complete_branch() -> None:
    result = conservative_directional_topology_transfer(
        upward_area=[0.0],
        upward_discharge=[0.0],
        downward_area=[1.5e-3],
        downward_discharge=[2.0e-5],
        preserve_stopped_partition=[True],
    )

    assert result.state.upward_area == pytest.approx([1.5e-3])
    assert result.state.upward_discharge == pytest.approx([2.0e-5])
    assert result.state.downward_area == (0.0,)
    assert result.state.downward_discharge == (0.0,)


def test_physical_drag_single_liquid_reduces_to_existing_pair_formula() -> None:
    params = _parameters(cells=1, dz=0.10)
    area_up = 1.0e-3
    liquid_velocity = 0.20
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[area_up],
        upward_discharge=[area_up * liquid_velocity],
        downward_area=[0.0],
        downward_discharge=[0.0],
    )
    gas_area = params.full_area - area_up
    rho_g = 1.8
    gas_mass = rho_g * gas_area * params.cell_length
    gas_velocity = 1.0
    perimeter = 0.055
    diameter = 0.030
    viscosity = 1.81e-5
    dt = 0.004
    gas = PhysicalGasInterphaseState.from_iterables(
        gas_mass=[gas_mass],
        gas_momentum=[gas_mass * gas_velocity],
        gas_area=[gas_area],
        upward_interface_perimeter=[perimeter],
        downward_interface_perimeter=[0.0],
        upward_hydraulic_diameter=[diameter],
        downward_hydraulic_diameter=[0.0],
        gas_viscosity=viscosity,
    )

    reynolds = rho_g * abs(gas_velocity - liquid_velocity) * diameter / viscosity
    friction = 16.0 / reynolds if reynolds < 2100.0 else 0.046 * reynolds**-0.2
    friction = min(max(friction, 0.0), 4.0)
    force_coefficient = 0.5 * friction * rho_g * perimeter * params.cell_length
    liquid_mass = params.liquid_density * area_up * params.cell_length
    relative = gas_velocity - liquid_velocity
    beta = force_coefficient * (1.0 / gas_mass + 1.0 / liquid_mass)
    relative_new = relative / (1.0 + beta * abs(relative) * dt)
    mixture_velocity = (
        gas_mass * gas_velocity + liquid_mass * liquid_velocity
    ) / (gas_mass + liquid_mass)
    expected_gas_velocity = (
        mixture_velocity
        + liquid_mass / (gas_mass + liquid_mass) * relative_new
    )
    expected_liquid_velocity = (
        mixture_velocity
        - gas_mass / (gas_mass + liquid_mass) * relative_new
    )

    result = implicit_physical_three_body_drag_exchange(
        state,
        params,
        gas,
        dt=dt,
    )

    assert result.gas_momentum[0] / gas_mass == pytest.approx(
        expected_gas_velocity, rel=2.0e-14
    )
    assert result.state.upward_discharge[0] / area_up == pytest.approx(
        expected_liquid_velocity, rel=2.0e-14
    )


def test_taylor_breakthrough_map_preserves_each_cell_area_and_momentum() -> None:
    params = _parameters(cells=3, dz=0.02)
    full = params.full_area
    area = [0.80 * full, 0.55 * full, 0.30 * full]
    discharge = [1.0e-4, -2.0e-5, 3.0e-5]

    result = map_taylor_breakthrough_to_twostream(
        area,
        discharge,
        params,
        taylor_core_area_fraction=0.76,
        taylor_rise_velocity=0.345 * (params.gravity * params.diameter) ** 0.5,
    )

    assert result.state.liquid_area == pytest.approx(area, abs=2.0e-18)
    assert result.state.liquid_discharge == pytest.approx(
        discharge, abs=2.0e-18
    )
    assert result.area_residual == pytest.approx([0.0] * 3, abs=2.0e-18)
    assert result.momentum_residual == pytest.approx([0.0] * 3, abs=2.0e-18)
    assert result.state.upward_area == pytest.approx(
        [area[0], 0.0, area[2]], abs=2.0e-18
    )
    assert result.state.downward_area == pytest.approx(
        [0.0, area[1], 0.0], abs=2.0e-18
    )
    assert result.state.upward_discharge == pytest.approx(
        [discharge[0], 0.0, discharge[2]], abs=2.0e-18
    )
    assert result.state.downward_discharge == pytest.approx(
        [0.0, discharge[1], 0.0], abs=2.0e-18
    )
    # The Davies--Taylor film speed is diagnostic at this topology event; it
    # is not projected onto cells whose inherited momentum is upward.
    assert result.state.downward_discharge[0] == 0.0
    assert result.state.downward_discharge[2] == 0.0
    assert result.falling_film_velocity < 0.0


def test_taylor_breakthrough_map_assigns_resting_old_water_to_downward_owner() -> None:
    params = _parameters(cells=2, dz=0.02)
    full = params.full_area
    result = map_taylor_breakthrough_to_twostream(
        [0.70 * full, 0.15 * full],
        [0.0, 0.0],
        params,
        taylor_core_area_fraction=0.80,
        taylor_rise_velocity=0.30,
        swept_fraction=[1.0, 0.5],
    )

    assert result.state.liquid_area == pytest.approx(
        [0.70 * full, 0.15 * full], abs=2.0e-18
    )
    assert result.state.upward_discharge == (0.0, 0.0)
    assert result.state.downward_discharge == (0.0, 0.0)
    assert result.state.upward_area == pytest.approx([0.0, 0.0], abs=2.0e-18)
    assert result.state.downward_area == pytest.approx(
        [0.70 * full, 0.15 * full], abs=2.0e-18
    )


def test_stopped_stream_is_conservatively_merged_instead_of_rejected() -> None:
    transfer = conservative_directional_topology_transfer(
        upward_area=[1.2e-3],
        upward_discharge=[0.0],
        downward_area=[0.8e-3],
        downward_discharge=[-2.0e-4],
    )

    assert transfer.state.upward_area == pytest.approx([0.0])
    assert transfer.state.downward_area == pytest.approx([2.0e-3])
    assert transfer.state.liquid_discharge == pytest.approx([-2.0e-4])
    assert transfer.area_residual == pytest.approx(0.0, abs=2.0e-18)
    assert transfer.momentum_residual == pytest.approx(0.0, abs=2.0e-18)
    assert transfer.kinematic_energy_loss >= 0.0


def test_geometric_mouth_can_preserve_a_stopped_upward_corridor() -> None:
    transfer = conservative_directional_topology_transfer(
        upward_area=[1.2e-3],
        upward_discharge=[0.0],
        downward_area=[0.8e-3],
        downward_discharge=[-2.0e-4],
        preserve_stopped_partition=[True],
    )

    assert transfer.state.upward_area == pytest.approx([1.2e-3])
    assert transfer.state.upward_discharge == (0.0,)
    assert transfer.state.downward_area == pytest.approx([0.8e-3])
    assert transfer.state.downward_discharge == pytest.approx([-2.0e-4])
    assert transfer.area_residual == pytest.approx(0.0, abs=2.0e-18)
    assert transfer.momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_crossed_streams_are_swapped_without_area_momentum_or_energy_loss() -> None:
    transfer = conservative_directional_topology_transfer(
        upward_area=[1.1e-3],
        upward_discharge=[-2.2e-4],
        downward_area=[0.7e-3],
        downward_discharge=[1.4e-4],
    )

    assert transfer.state.upward_area == pytest.approx([0.7e-3])
    assert transfer.state.upward_discharge == pytest.approx([1.4e-4])
    assert transfer.state.downward_area == pytest.approx([1.1e-3])
    assert transfer.state.downward_discharge == pytest.approx([-2.2e-4])
    assert transfer.area_residual == pytest.approx(0.0, abs=2.0e-18)
    assert transfer.momentum_residual == pytest.approx(0.0, abs=2.0e-18)
    assert transfer.kinematic_energy_loss == pytest.approx(0.0, abs=2.0e-18)


def test_fv_stage_applies_stopped_stream_topology_transfer_conservatively() -> None:
    params = _parameters(cells=1, dz=0.20)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3],
        upward_discharge=[0.0],
        downward_area=[1.0e-3],
        downward_discharge=[-1.0e-4],
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
    )

    assert result.state.upward_area == pytest.approx([0.0])
    assert result.state.downward_area == pytest.approx([2.0e-3])
    assert result.ledger.upward_topology_volume_transfer < 0.0
    assert result.ledger.downward_topology_volume_transfer > 0.0
    assert result.ledger.total_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert (
        result.ledger.topology_upward_momentum_transfer
        + result.ledger.topology_downward_momentum_transfer
    ) == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_large_outflow_is_limited_by_the_actual_donor_inventory() -> None:
    params = _parameters(cells=1, dz=1.0)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-4],
        upward_discharge=[1.0e-2],
        downward_area=[1.0e-4],
        downward_discharge=[-1.0e-2],
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            downward_rate=1.0e-2,
            downward_speed=100.0,
        ),
        top=DirectionalBoundaryFlux(
            upward_rate=1.0e-2,
            upward_speed=100.0,
        ),
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.10,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert result.upward_donor_factor[0] == pytest.approx(0.1)
    assert result.downward_donor_factor[0] == pytest.approx(0.1)
    assert result.state.upward_area[0] == pytest.approx(0.0, abs=2.0e-18)
    assert result.state.downward_area[0] == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.total_volume_residual == pytest.approx(0.0, abs=2.0e-18)


def test_counter_current_arrivals_share_receiver_capacity_conservatively() -> None:
    """Two legal donor fluxes may not jointly over-fill their receiver."""

    params = _parameters(cells=3, dz=0.10)
    full = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.45 * full] * 3,
        upward_discharge=[4.0e-3, 1.0e-4, 1.0e-4],
        downward_area=[0.35 * full, 0.45 * full, 0.35 * full],
        downward_discharge=[-1.0e-4, -1.0e-4, -4.0e-3],
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=1.0e-4 / (0.45 * full),
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / (0.35 * full),
        ),
        top=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=1.0e-4 / (0.45 * full),
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / (0.35 * full),
        ),
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.02,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert result.ledger.capacity_projection_iterations > 0
    assert min(result.upward_receiving_factor) < 1.0
    assert min(result.downward_receiving_factor) < 1.0
    assert max(result.state.liquid_area) <= full + params.packing_tolerance
    assert result.ledger.upward_volume_residual == pytest.approx(0.0, abs=3.0e-18)
    assert result.ledger.downward_volume_residual == pytest.approx(0.0, abs=3.0e-18)
    assert result.ledger.capacity_boundary_volume_residual == pytest.approx(
        0.0, abs=3.0e-18
    )
    # Interior face corrections telescope; no directional volume disappears.
    assert sum(result.upward_capacity_volume_correction) == pytest.approx(
        result.ledger.upward_capacity_boundary_volume_change,
        abs=3.0e-18,
    )
    assert sum(result.downward_capacity_volume_correction) == pytest.approx(
        result.ledger.downward_capacity_boundary_volume_change,
        abs=3.0e-18,
    )
    assert (
        sum(result.upward_capacity_momentum_impulse)
        + sum(result.downward_capacity_momentum_impulse)
    ) == pytest.approx(
        result.ledger.capacity_constraint_momentum_impulse,
        abs=3.0e-18,
    )
    assert result.ledger.capacity_momentum_ledger_residual == pytest.approx(
        0.0, abs=3.0e-18
    )


@pytest.mark.parametrize(
    ("cells", "dz", "dt"),
    [(3, 0.10, 0.020), (5, 0.06, 0.012), (9, 0.03, 0.006)],
)
def test_receiver_projection_is_stable_across_reasonable_grids_and_steps(
    cells: int,
    dz: float,
    dt: float,
) -> None:
    params = _parameters(cells=cells, dz=dz)
    full = params.full_area
    middle = cells // 2
    upward_area = [0.40 * full] * cells
    downward_area = [0.40 * full] * cells
    upward_area[middle] = 0.45 * full
    downward_area[middle] = 0.45 * full
    upward_q = [1.0e-4] * cells
    downward_q = [-1.0e-4] * cells
    upward_q[middle - 1] = 4.0e-3
    downward_q[middle + 1] = -4.0e-3
    state = VerticalTwoStreamState.from_iterables(
        upward_area=upward_area,
        upward_discharge=upward_q,
        downward_area=downward_area,
        downward_discharge=downward_q,
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=1.0e-4 / upward_area[0],
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / downward_area[0],
        ),
        top=DirectionalBoundaryFlux(
            upward_rate=1.0e-4,
            upward_speed=1.0e-4 / upward_area[-1],
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / downward_area[-1],
        ),
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert result.ledger.capacity_projection_iterations > 0
    assert max(result.state.liquid_area) <= full + params.packing_tolerance
    assert result.ledger.upward_volume_residual == pytest.approx(0.0, abs=4.0e-18)
    assert result.ledger.downward_volume_residual == pytest.approx(0.0, abs=4.0e-18)
    assert result.ledger.liquid_momentum_residual == pytest.approx(0.0, abs=4.0e-18)


def test_receiving_projection_is_an_exact_noop_when_capacity_is_inactive() -> None:
    params = _parameters(cells=3, dz=0.10)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.7e-3] * 3,
        upward_discharge=[2.0e-4] * 3,
        downward_area=[0.8e-3] * 3,
        downward_discharge=[-1.0e-4] * 3,
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=2.0e-4,
            upward_speed=2.0e-4 / 1.7e-3,
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / 0.8e-3,
        ),
        top=DirectionalBoundaryFlux(
            upward_rate=2.0e-4,
            upward_speed=2.0e-4 / 1.7e-3,
            downward_rate=1.0e-4,
            downward_speed=1.0e-4 / 0.8e-3,
        ),
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.02,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert result.state == state
    assert result.upward_area_flux == pytest.approx([2.0e-4] * 4)
    assert result.downward_area_flux == pytest.approx([-1.0e-4] * 4)
    assert result.upward_receiving_factor == pytest.approx([1.0] * 4)
    assert result.downward_receiving_factor == pytest.approx([1.0] * 4)
    assert result.upward_capacity_volume_correction == pytest.approx([0.0] * 3)
    assert result.downward_capacity_volume_correction == pytest.approx([0.0] * 3)
    assert result.ledger.capacity_projection_iterations == 0
    assert result.ledger.capacity_constraint_momentum_impulse == pytest.approx(0.0)
    assert result.ledger.capacity_momentum_ledger_residual == pytest.approx(0.0)


def test_bottom_net_capacity_includes_storage_and_resolved_throughflow() -> None:
    params = _parameters(cells=2, dz=0.10)
    area = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.90 * area, 0.50 * area],
        upward_discharge=[0.03 * area, 0.0],
        downward_area=[0.0, 0.40 * area],
        downward_discharge=[0.0, 0.0],
    )

    capacity = bottom_net_receiving_rate_capacity(
        state,
        params,
        dt=0.10,
    )

    # Cell 0 can store 0.10*A over dt/dz=1 and pass 0.03*A upward.
    assert capacity == pytest.approx(
        0.13 * area + params.packing_tolerance,
        abs=1.0e-15,
    )
    boundary = DirectionalBoundaryFlux(
        upward_rate=capacity,
        upward_speed=capacity / (0.90 * area),
    )
    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.10,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(bottom=boundary),
    )
    assert result.upward_area_flux[0] == pytest.approx(capacity)
    assert result.downward_area_flux[0] == 0.0


def test_full_stagnant_bottom_cell_has_zero_net_receiving_capacity() -> None:
    params = _parameters(cells=2, dz=0.10)
    area = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.80 * area, 0.30 * area],
        upward_discharge=[0.0, 0.0],
        downward_area=[0.20 * area, 0.20 * area],
        downward_discharge=[0.0, 0.0],
    )

    capacity = bottom_net_receiving_rate_capacity(
        state,
        params,
        dt=0.10,
    )

    assert capacity == pytest.approx(
        params.packing_tolerance,
        abs=1.0e-15,
    )


def test_open_mouth_capacity_preserves_gas_corridor_and_throughflow() -> None:
    params = _parameters(cells=2, dz=0.10)
    area = params.full_area
    mouth_liquid_capacity = 0.50 * area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.10 * area, 0.10 * area],
        upward_discharge=[0.02 * area, 0.0],
        downward_area=[0.40 * area, 0.20 * area],
        downward_discharge=[0.0, 0.0],
    )
    capacity_area = [mouth_liquid_capacity, area]

    capacity = bottom_net_receiving_rate_capacity(
        state,
        params,
        dt=0.10,
        liquid_capacity_area=capacity_area,
    )

    # The already open half-area gas corridor removes storage capacity, but
    # the resolved upward throughflow may still be replaced at the mouth.
    assert capacity == pytest.approx(
        0.02 * area + params.packing_tolerance,
        abs=1.0e-15,
    )
    boundary = DirectionalBoundaryFlux(
        upward_rate=capacity,
        upward_speed=capacity / (0.10 * area),
    )
    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.10,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(bottom=boundary),
        liquid_capacity_area=capacity_area,
    )
    assert (
        result.state.liquid_area[0]
        <= mouth_liquid_capacity + params.packing_tolerance + 1.0e-15
    )
    assert result.upward_area_flux[0] == pytest.approx(capacity)


def test_capacity_iterables_are_materialized_once() -> None:
    params = _parameters(cells=2, dz=0.10)
    area = params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.20 * area, 0.20 * area],
        upward_discharge=[0.0, 0.0],
        downward_area=[0.30 * area, 0.30 * area],
        downward_discharge=[0.0, 0.0],
    )
    capacity = [0.60 * area, 0.70 * area]

    from_list = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
        liquid_capacity_area=capacity,
    )
    from_generator = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
        liquid_capacity_area=(value for value in capacity),
    )

    assert from_generator == from_list


def test_unsaturated_preserved_corridor_survives_pressure_sign_crossing() -> None:
    params = _parameters(cells=1, dz=0.10)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[1.0e-3],
        upward_discharge=[1.0e-4],
        downward_area=[8.0e-4],
        downward_discharge=[-8.0e-5],
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=[130_000.0, 100_000.0],
        preserve_stopped_partition=[True],
    )

    assert result.state.liquid_area == pytest.approx(state.liquid_area)
    assert result.state.upward_discharge[0] >= 0.0
    assert result.state.downward_discharge[0] <= 0.0
    assert result.state.downward_area[0] > 0.0
    assert result.ledger.liquid_momentum_residual == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_tolerance_scale_counterflow_chain_is_projected_in_one_finite_pass() -> None:
    """Regress the 3.294334e-12 m2 residual that stalled the old iteration."""

    cells = 20
    dz = 0.02
    dt = 0.005
    params = _parameters(cells=cells, dz=dz)
    full = params.full_area
    reported_area_excess = 3.294334e-12
    directional_rate = reported_area_excess / (dt / dz)
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.55 * full] * cells,
        upward_discharge=[directional_rate] * cells,
        downward_area=[0.45 * full] * cells,
        downward_discharge=[-directional_rate] * cells,
    )
    boundaries = VerticalTwoStreamBoundaries(
        bottom=DirectionalBoundaryFlux(
            upward_rate=directional_rate,
            upward_speed=directional_rate / (0.55 * full),
        ),
        top=DirectionalBoundaryFlux(
            downward_rate=directional_rate,
            downward_speed=directional_rate / (0.45 * full),
        ),
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=boundaries,
    )

    assert result.ledger.capacity_projection_iterations == 1
    assert max(result.state.liquid_area) <= full + params.packing_tolerance
    assert result.ledger.maximum_packing_residual <= 0.0
    assert result.ledger.upward_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.downward_volume_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.capacity_boundary_volume_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.ledger.capacity_momentum_ledger_residual == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_degenerate_one_stream_matches_the_legacy_area_and_discharge() -> None:
    params = _parameters(cells=4, dz=0.10)
    legacy_area = [2.0e-3] * 4
    legacy_discharge = [2.0e-4] * 4
    state = VerticalTwoStreamState.from_legacy_single_stream(
        legacy_area,
        legacy_discharge,
    )
    throughflow = DirectionalBoundaryFlux(
        upward_rate=2.0e-4,
        upward_speed=0.10,
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.02,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(
            bottom=throughflow,
            top=throughflow,
        ),
    )

    assert result.state.liquid_area == pytest.approx(legacy_area)
    assert result.state.liquid_discharge == pytest.approx(legacy_discharge)
    assert result.state.downward_area == pytest.approx([0.0] * 4)
    assert result.state.downward_discharge == pytest.approx([0.0] * 4)
    assert result.ledger.total_volume_residual == pytest.approx(0.0, abs=2.0e-18)


def test_accepted_face_does_not_algebraically_reset_packed_cell_momentum() -> None:
    """Without an explicit liquid cap, no capacity pressure is introduced."""

    params = _parameters(cells=3, dz=0.10)
    area = params.full_area
    accepted_rate = 3.0e-4
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.0, 0.0, 0.0],
        upward_discharge=[0.0, 0.0, 0.0],
        downward_area=[area, area, area],
        downward_discharge=[-1.0e-2, -2.0e-3, -8.0e-3],
    )
    throughflow = DirectionalBoundaryFlux(
        downward_rate=accepted_rate,
        downward_speed=accepted_rate / area,
    )

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=0.01,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(
            bottom=throughflow,
            top=throughflow,
        ),
    )

    assert result.downward_area_flux == pytest.approx(
        [-accepted_rate] * 4
    )
    assert result.state.liquid_discharge[0] != pytest.approx(-accepted_rate)
    assert result.capacity_pressure_cell_impulse == pytest.approx([0.0] * 3)
    assert result.capacity_pressure_face_momentum_flux == pytest.approx([0.0] * 4)
    assert result.ledger.capacity_pressure_kinematic_impulse == pytest.approx(0.0)


def test_explicit_capacity_projects_donor_block_without_a_bulk_face_anchor() -> None:
    """Formal pressure enforces packing but does not equate cell 0 to its face."""

    params = _parameters(cells=3, dz=0.10)
    area = params.full_area
    accepted_rate = 3.0e-4
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.0, 0.0, 0.0],
        upward_discharge=[0.0, 0.0, 0.0],
        downward_area=[area, area, area],
        downward_discharge=[-1.0e-2, -2.0e-3, -8.0e-3],
    )
    throughflow = DirectionalBoundaryFlux(
        downward_rate=accepted_rate,
        downward_speed=accepted_rate / area,
    )
    dt = 0.01

    result = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(
            bottom=throughflow,
            top=throughflow,
        ),
        liquid_capacity_area=[area] * 3,
        preserve_stopped_partition=[True] * 3,
        enable_capacity_pressure_projection=True,
    )

    assert result.state.downward_discharge[0] != pytest.approx(-accepted_rate)
    assert result.state.downward_discharge[1:] == pytest.approx(
        [-accepted_rate] * 2
    )
    assert any(abs(value) > 0.0 for value in result.capacity_pressure_cell_impulse)
    face_flux = result.capacity_pressure_face_momentum_flux
    for cell, impulse in enumerate(result.capacity_pressure_cell_impulse):
        assert dt / params.cell_length * (
            face_flux[cell] - face_flux[cell + 1]
        ) == pytest.approx(impulse)
    ledger = result.ledger
    assert ledger.capacity_pressure_bottom_bulk_anchor_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert ledger.capacity_pressure_physical_impulse == pytest.approx(
        params.liquid_density
        * params.cell_length
        * sum(result.capacity_pressure_cell_impulse)
    )
    assert ledger.capacity_pressure_decomposition_residual == pytest.approx(
        0.0, abs=1.0e-14
    )
    assert ledger.capacity_pressure_coupled_momentum_residual == pytest.approx(
        0.0, abs=1.0e-14
    )
    assert ledger.liquid_momentum_residual == pytest.approx(0.0, abs=2.0e-18)


def test_physical_turn_reaction_acts_only_on_falling_donor() -> None:
    """The caller excludes structural rejection from this liquid traction."""

    params = _parameters(cells=1, dz=0.10)
    area = 0.45 * params.full_area
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[area],
        upward_discharge=[1.0e-4],
        downward_area=[area],
        downward_discharge=[-1.0e-4],
    )
    boundary = DirectionalBoundaryFlux(
        upward_rate=1.0e-4,
        upward_speed=1.0e-4 / area,
        downward_rate=1.0e-4,
        downward_speed=1.0e-4 / area,
    )
    dt = 0.01
    reaction_flux = 2.0e-4
    without = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(bottom=boundary, top=boundary),
    )
    with_reaction = advance_vertical_two_stream_fv(
        state,
        params,
        dt=dt,
        pressure_faces=_hydrostatic(params),
        boundaries=VerticalTwoStreamBoundaries(bottom=boundary, top=boundary),
        bottom_downward_reaction_flux=reaction_flux,
    )

    assert with_reaction.state.upward_discharge[0] == pytest.approx(
        without.state.upward_discharge[0]
    )
    expected = dt / params.cell_length * reaction_flux
    assert with_reaction.state.downward_discharge[0] - without.state.downward_discharge[0] == pytest.approx(
        expected
    )
    assert with_reaction.ledger.bottom_downward_reaction_momentum_impulse == pytest.approx(
        dt * reaction_flux
    )
    assert with_reaction.ledger.liquid_momentum_residual == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_total_and_film_mapping_recovers_existing_alr_and_net_discharge() -> None:
    state = VerticalTwoStreamState.from_total_and_film(
        liquid_area=[3.0e-3, 2.8e-3],
        liquid_discharge=[1.0e-4, -2.0e-5],
        falling_film_area=[1.0e-3, 1.2e-3],
        falling_film_discharge=[-1.5e-4, -1.1e-4],
    )

    assert state.liquid_area == pytest.approx([3.0e-3, 2.8e-3])
    assert state.liquid_discharge == pytest.approx([1.0e-4, -2.0e-5])
    assert state.gross_upward_flow == pytest.approx([2.5e-4, 9.0e-5])
    assert state.gross_downward_flow == pytest.approx([1.5e-4, 1.1e-4])


def test_scope_is_explicit_and_step_has_no_result_feedback_inputs() -> None:
    assert TWOSTREAM_FV_CORE_READY is True
    assert COMPLETE_CASEA_RISER_READY is False
    assert "finite_tjunction_two_stream_riemann_problem" in MISSING_PHYSICAL_CLOSURES
    signature = inspect.signature(advance_vertical_two_stream_fv)
    assert set(signature.parameters) == {
        "state",
        "parameters",
        "dt",
        "pressure_faces",
        "boundaries",
        "gas_coupling",
        "preserve_stopped_partition",
            "liquid_capacity_area",
            "bottom_downward_reaction_flux",
            "enable_capacity_pressure_projection",
        }
