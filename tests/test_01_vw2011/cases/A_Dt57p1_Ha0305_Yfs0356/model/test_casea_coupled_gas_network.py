from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from casea_coupled_gas_network import (  # noqa: E402
    CoupledGasParameters,
    OpenIsothermalGasInventory,
    _apply_downstream_material_front_kinematics,
    _apply_side_t_phase_separation,
    _equilibrate_horizontal_front_receivers,
    _equilibrate_vertical_front_receivers,
    _implicit_interphase_drag_exchange,
    _mass_backed_gas_topology,
    advance_lumped_pocket_vertical_network,
    advance_lumped_isothermal_side_t,
    advance_coupled_gas_network,
    isothermal_ideal_gas_riemann_flux,
    junction_mouth_area,
)


def _uniform_lumped_vertical_state(
    params: CoupledGasParameters,
    *,
    cells: int = 12,
    dz: float = 0.02,
    liquid_fraction: float = 0.55,
) -> tuple[np.ndarray, ...]:
    liquid_area = np.full(
        cells,
        liquid_fraction * params.vertical_area,
    )
    gas_area = params.vertical_area - liquid_area
    gas_mass = params.rho_atmospheric * gas_area * dz
    return (
        gas_mass,
        np.zeros(cells),
        np.zeros(cells),
        liquid_area,
        np.zeros(cells),
    )


def test_lumped_pocket_vertical_equal_pressure_has_zero_t_flux() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    dz = 0.02
    state = _uniform_lumped_vertical_state(params, dz=dz)
    void_area = 0.45 * params.horizontal_area
    pocket_volume = void_area * 0.18
    inventory = OpenIsothermalGasInventory(
        mass=params.rho_atmospheric * pocket_volume,
        volume=pocket_volume,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
    )
    result = advance_lumped_pocket_vertical_network(
        inventory,
        void_area,
        *state,
        dz=dz,
        dt=4.0e-4,
        params=params,
    )
    assert abs(result.junction_mass_transfer) < 2.0e-15
    assert math.isclose(
        result.horizontal_inventory.mass,
        inventory.mass,
        rel_tol=0.0,
        abs_tol=2.0e-15,
    )
    np.testing.assert_allclose(
        result.vertical_total_mass,
        state[0],
        rtol=0.0,
        atol=2.0e-15,
    )
    assert abs(result.total_mass_error) < 2.0e-13


def test_lumped_pocket_vertical_pressure_drives_forward_blowdown() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    dz = 0.02
    state = _uniform_lumped_vertical_state(params, dz=dz)
    void_area = 0.45 * params.horizontal_area
    pocket_volume = void_area * 0.18
    inventory = OpenIsothermalGasInventory(
        mass=1.025 * params.rho_atmospheric * pocket_volume,
        volume=pocket_volume,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
    )
    result = advance_lumped_pocket_vertical_network(
        inventory,
        void_area,
        *state,
        dz=dz,
        dt=4.0e-4,
        params=params,
    )
    assert result.junction_mouth_area > 0.0
    assert result.junction_mass_transfer > 0.0
    assert result.horizontal_inventory.mass < inventory.mass
    assert result.horizontal_inventory.pressure_absolute < inventory.pressure_absolute
    assert float(np.sum(result.vertical_tracer_mass)) > 0.0
    assert math.isclose(
        inventory.mass - result.horizontal_inventory.mass,
        result.junction_mass_transfer,
        rel_tol=0.0,
        abs_tol=2.0e-13,
    )
    assert abs(result.total_mass_error) < 2.0e-11
    assert abs(result.tracer_mass_error) < 2.0e-11


def test_lumped_pocket_vertical_reports_mass_and_escape_ledgers() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    dz = 0.02
    state = list(_uniform_lumped_vertical_state(params, cells=8, dz=dz))
    state[1][-3:] = state[0][-3:] * 2.0
    state[2][-3:] = 0.60 * state[0][-3:]
    void_area = 0.45 * params.horizontal_area
    pocket_volume = void_area * 0.18
    inventory = OpenIsothermalGasInventory(
        mass=params.rho_atmospheric * pocket_volume,
        volume=pocket_volume,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
    )
    total_before = inventory.mass + math.fsum(state[0])
    tracer_before = inventory.mass + math.fsum(state[2])
    result = advance_lumped_pocket_vertical_network(
        inventory,
        void_area,
        *state,
        dz=dz,
        dt=8.0e-4,
        params=params,
    )
    total_after = (
        result.horizontal_inventory.mass
        + math.fsum(result.vertical_total_mass)
        + result.atmospheric_mass_exchange
    )
    tracer_after = (
        result.horizontal_inventory.mass
        + math.fsum(result.vertical_tracer_mass)
        + result.escaped_tracer_mass
    )
    assert result.escaped_tracer_mass > 0.0
    assert math.isclose(total_after, total_before, rel_tol=0.0, abs_tol=2.0e-11)
    assert math.isclose(
        tracer_after - tracer_before,
        result.tracer_mass_error,
        rel_tol=0.0,
        abs_tol=2.0e-18,
    )
    assert abs(result.total_mass_error) < 2.0e-11
    # With an imposed top outflow pulse the acoustic solution briefly reverses
    # at the T.  The legacy network treats every horizontal molecule as pocket
    # tracer, so reverse atmospheric entrainment produces a small, explicitly
    # reported tracer-ledger defect.  Total gas mass remains exactly closed.
    assert abs(result.tracer_mass_error) < 5.0e-10


def test_open_atmospheric_core_equalizes_with_gas_not_liquid_head() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=9.81,
    )
    cells = 12
    dz = 0.02
    # A 98%-gas open core with only a falling wall-film inventory.  Once this
    # component reaches the atmospheric top, the gas must not inherit a full
    # rho_l*g pressure gradient from the residual film.
    liquid_area = np.full(cells, 0.02 * params.vertical_area)
    gas_area = params.vertical_area - liquid_area
    vertical_mass = params.rho_atmospheric * gas_area * dz
    vertical_momentum = np.zeros(cells)
    vertical_tracer = np.zeros(cells)
    vertical_liquid_q = np.zeros(cells)
    horizontal_void_area = 0.90 * params.horizontal_area
    pocket_volume = 3.5e-3
    inventory = OpenIsothermalGasInventory(
        mass=0.94 * params.rho_atmospheric * pocket_volume,
        volume=pocket_volume,
        gas_constant=params.gas_constant,
        temperature=params.gas_temperature,
    )
    mass_initial = inventory.mass + math.fsum(vertical_mass)
    atmospheric_exchange = 0.0

    for _ in range(80):
        result = advance_lumped_pocket_vertical_network(
            inventory,
            horizontal_void_area,
            vertical_mass,
            vertical_momentum,
            vertical_tracer,
            liquid_area,
            vertical_liquid_q,
            dz=dz,
            dt=1.0e-3,
            params=params,
        )
        inventory = result.horizontal_inventory
        vertical_mass = result.vertical_total_mass
        vertical_momentum = result.vertical_momentum
        vertical_tracer = result.vertical_tracer_mass
        atmospheric_exchange += result.atmospheric_mass_exchange

    pressure_ratio = inventory.pressure_absolute / params.atmospheric_pressure
    assert abs(pressure_ratio - 1.0) < 2.0e-4
    assert inventory.mass > 0.94 * params.rho_atmospheric * pocket_volume
    assert atmospheric_exchange < 0.0
    assert math.isclose(
        inventory.mass + math.fsum(vertical_mass) + atmospheric_exchange,
        mass_initial,
        rel_tol=0.0,
        abs_tol=2.0e-12,
    )


def test_open_isothermal_inventory_pressure_is_state_derived() -> None:
    inventory = OpenIsothermalGasInventory(
        mass=0.012,
        volume=0.010,
        gas_constant=287.05,
        temperature=293.0,
    )
    expected = inventory.mass * 287.05 * 293.0 / inventory.volume
    assert math.isclose(
        inventory.pressure_absolute,
        expected,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    expanded = inventory.with_state(volume=2.0 * inventory.volume)
    assert expanded.mass == inventory.mass
    assert math.isclose(
        expanded.pressure_absolute,
        0.5 * inventory.pressure_absolute,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_equal_pressure_stationary_t_states_have_zero_mass_flux() -> None:
    density = 1.18
    horizontal = OpenIsothermalGasInventory(
        mass=density * 0.012,
        volume=0.012,
    )
    vertical_volume = 7.5e-4
    vertical = np.array([density * vertical_volume, 3.0e-4])
    result = advance_lumped_isothermal_side_t(
        horizontal,
        vertical,
        vertical_receiver_volume=vertical_volume,
        mouth_area=2.0e-4,
        dt=2.0e-3,
    )
    assert result.mass_flux == 0.0
    assert result.raw_mass_transfer == 0.0
    assert result.mass_transfer == 0.0
    assert result.horizontal_inventory.mass == horizontal.mass
    np.testing.assert_array_equal(result.vertical_mass, vertical)
    assert result.conservation_error == 0.0

    direct_flux = isothermal_ideal_gas_riemann_flux(
        density,
        0.0,
        density,
        0.0,
        gas_constant=horizontal.gas_constant,
        temperature=horizontal.temperature,
    )
    assert direct_flux[0] == 0.0
    assert math.isclose(
        direct_flux[1],
        horizontal.pressure_absolute,
        rel_tol=5.0e-16,
        abs_tol=0.0,
    )


def test_lumped_side_t_riemann_flux_reverses_with_pressure_difference() -> None:
    vertical_volume = 1.0e-3
    mouth_area = 1.5e-4
    dt = 5.0e-4

    high_horizontal = OpenIsothermalGasInventory(
        mass=1.35 * 0.010,
        volume=0.010,
    )
    vertical_low = np.array([1.00 * vertical_volume, 2.0e-4])
    forward = advance_lumped_isothermal_side_t(
        high_horizontal,
        vertical_low,
        vertical_receiver_volume=vertical_volume,
        mouth_area=mouth_area,
        dt=dt,
    )
    assert forward.mass_flux > 0.0
    assert forward.mass_transfer > 0.0
    assert forward.horizontal_inventory.mass < high_horizontal.mass
    assert forward.vertical_mass[0] > vertical_low[0]
    assert abs(forward.conservation_error) < 2.0e-18

    low_horizontal = OpenIsothermalGasInventory(
        mass=0.95 * 0.010,
        volume=0.010,
    )
    vertical_high = np.array([1.30 * vertical_volume, 2.0e-4])
    reverse = advance_lumped_isothermal_side_t(
        low_horizontal,
        vertical_high,
        vertical_receiver_volume=vertical_volume,
        mouth_area=mouth_area,
        dt=dt,
    )
    assert reverse.mass_flux < 0.0
    assert reverse.mass_transfer < 0.0
    assert reverse.horizontal_inventory.mass > low_horizontal.mass
    assert reverse.vertical_mass[0] < vertical_high[0]
    assert abs(reverse.conservation_error) < 2.0e-18


def test_lumped_side_t_donor_bound_is_positive_and_strictly_conservative() -> None:
    vertical_volume = 1.0e-3

    horizontal_donor = OpenIsothermalGasInventory(
        mass=1.60e-3,
        volume=1.0e-3,
    )
    vertical = np.array([0.90e-3, 0.25e-3])
    forward = advance_lumped_isothermal_side_t(
        horizontal_donor,
        vertical,
        vertical_receiver_volume=vertical_volume,
        mouth_area=5.0e-4,
        dt=1.0e3,
    )
    assert forward.donor_limited
    assert forward.horizontal_inventory.mass == 0.0
    assert np.all(forward.vertical_mass >= 0.0)
    assert math.isclose(
        forward.horizontal_inventory.mass + math.fsum(forward.vertical_mass),
        horizontal_donor.mass + math.fsum(vertical),
        rel_tol=0.0,
        abs_tol=2.0e-18,
    )

    horizontal_receiver = OpenIsothermalGasInventory(
        mass=0.75e-3,
        volume=1.0e-3,
    )
    vertical_donor = np.array([1.55e-3, 0.25e-3])
    reverse = advance_lumped_isothermal_side_t(
        horizontal_receiver,
        vertical_donor,
        vertical_receiver_volume=vertical_volume,
        mouth_area=5.0e-4,
        dt=1.0e3,
    )
    assert reverse.donor_limited
    assert reverse.vertical_mass[0] == 0.0
    assert reverse.horizontal_inventory.mass >= 0.0
    assert math.isclose(
        reverse.horizontal_inventory.mass + math.fsum(reverse.vertical_mass),
        horizontal_receiver.mass + math.fsum(vertical_donor),
        rel_tol=0.0,
        abs_tol=2.0e-18,
    )


def test_open_riser_receives_new_side_t_gas_before_liquid_full_dead_leg() -> None:
    supported = np.array([True, True, True, False, False])
    receiver = np.array([False, False, False, True, False])
    separated = _apply_side_t_phase_separation(
        receiver,
        supported,
        junction_index=2,
        vertical_branch_receiving=True,
    )
    assert not separated[3]

    already_connected = _apply_side_t_phase_separation(
        receiver,
        np.array([True, True, True, True, False]),
        junction_index=2,
        vertical_branch_receiving=True,
    )
    assert already_connected[3]

    vertical_closed = _apply_side_t_phase_separation(
        receiver,
        supported,
        junction_index=2,
        vertical_branch_receiving=False,
    )
    assert vertical_closed[3]


def test_downstream_gas_front_requires_liquid_interface_motion() -> None:
    supported = np.array([True, True, True, False, False])
    receiver = np.array([False, False, False, True, False])
    area = np.ones(5)

    stopped = _apply_downstream_material_front_kinematics(
        receiver,
        supported,
        area,
        np.zeros(5),
        junction_index=2,
    )
    assert not stopped[3]

    advancing = _apply_downstream_material_front_kinematics(
        receiver,
        supported,
        area,
        np.array([0.0, 0.0, 0.20, 0.10, 0.0]),
        junction_index=2,
    )
    assert advancing[3]

    reversing = _apply_downstream_material_front_kinematics(
        receiver,
        supported,
        area,
        np.array([0.0, 0.0, -0.20, -0.10, 0.0]),
        junction_index=2,
    )
    assert not reversing[3]


def test_new_horizontal_front_cell_is_not_initialised_as_vacuum() -> None:
    mass = np.array([1.2, 0.01, 0.0])
    momentum = np.array([0.24, 0.0, 0.0])
    area = np.array([0.8, 0.4, 0.0])
    supported = np.array([True, False, False])
    receiver = np.array([False, True, False])
    mass_before = float(np.sum(mass))
    momentum_before = float(np.sum(momentum))
    remapped_mass, remapped_momentum = (
        _equilibrate_horizontal_front_receivers(
            mass,
            momentum,
            area,
            supported,
            receiver,
            cell_width=0.05,
        )
    )
    np.testing.assert_allclose(np.sum(remapped_mass), mass_before, atol=1.0e-15)
    np.testing.assert_allclose(
        np.sum(remapped_momentum), momentum_before, atol=1.0e-15
    )
    density = remapped_mass[:2] / (area[:2] * 0.05)
    velocity = remapped_momentum[:2] / remapped_mass[:2]
    np.testing.assert_allclose(density[0], density[1], atol=1.0e-14)
    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-14)


def test_new_vertical_front_cell_is_filled_conservatively_from_tee() -> None:
    horizontal_mass = np.array([0.02, 0.08, 0.01])
    horizontal_momentum = np.array([0.0, 0.024, 0.0])
    vertical_mass = np.array([0.001, 0.004, 0.006])
    vertical_momentum = np.array([0.0002, 0.0004, 0.0])
    vertical_tracer = np.array([0.0, 0.004, 0.0])
    horizontal_area = np.array([0.2, 0.4, 0.1])
    vertical_area = np.array([0.3, 0.2, 0.1])
    supported = np.array([False, True, False])
    receiver = np.array([True, False, False])
    gas_mass_before = float(np.sum(horizontal_mass) + np.sum(vertical_mass))
    tracer_before = float(np.sum(horizontal_mass) + np.sum(vertical_tracer))
    horizontal_velocity_before = (
        horizontal_momentum[1] / horizontal_mass[1]
    )
    vertical_momentum_before = vertical_momentum.copy()

    hm, hj, vm, vj, vc = _equilibrate_vertical_front_receivers(
        horizontal_mass,
        horizontal_momentum,
        vertical_mass,
        vertical_momentum,
        vertical_tracer,
        horizontal_area,
        vertical_area,
        np.array([True, True, False]),
        supported,
        receiver,
        junction_index=1,
        horizontal_width=0.05,
        vertical_width=0.04,
    )

    np.testing.assert_allclose(
        np.sum(hm) + np.sum(vm), gas_mass_before, atol=1.0e-15
    )
    np.testing.assert_allclose(
        np.sum(hm) + np.sum(vc), tracer_before, atol=1.0e-15
    )
    donor_density = (
        np.sum(hm[:2]) / (np.sum(horizontal_area[:2]) * 0.05)
    )
    receiver_density = vm[0] / (vertical_area[0] * 0.04)
    np.testing.assert_allclose(donor_density, receiver_density, atol=1.0e-14)
    np.testing.assert_allclose(hj[1] / hm[1], horizontal_velocity_before)
    np.testing.assert_allclose(vj, vertical_momentum_before)
    assert vc[0] > 0.0
    assert np.all(vc >= 0.0)
    assert np.all(vc <= vm)


def test_vertical_front_initializer_never_relabels_reverse_gas_as_tracer() -> None:
    hm0 = np.array([1.0])
    vm0 = np.array([2.0])
    vc0 = np.array([0.5])
    hm, hj, vm, vj, vc = _equilibrate_vertical_front_receivers(
        hm0,
        np.array([0.2]),
        vm0,
        np.array([0.3]),
        vc0,
        np.array([1.0]),
        np.array([1.0]),
        np.array([True]),
        np.array([False]),
        np.array([True]),
        junction_index=0,
        horizontal_width=1.0,
        vertical_width=1.0,
    )
    np.testing.assert_allclose(hm, hm0)
    np.testing.assert_allclose(vm, vm0)
    np.testing.assert_allclose(vc, vc0)
    np.testing.assert_allclose(hj, 0.2)
    np.testing.assert_allclose(vj, 0.3)


def test_compressed_narrow_gas_cell_bridges_only_an_active_component() -> None:
    raw = np.array([0.20, 2.9e-4, 2.9e-4, 2.9e-4])
    rho = 1.2
    mass = rho * raw
    supported = _mass_backed_gas_topology(
        raw,
        mass,
        full_area=1.0,
        cell_width=1.0,
        rho_reference=rho,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    np.testing.assert_array_equal(supported, [True, True, True, True])

    isolated = _mass_backed_gas_topology(
        np.array([0.0, 2.9e-4, 0.0]),
        np.array([0.0, rho * 2.9e-4, 0.0]),
        full_area=1.0,
        cell_width=1.0,
        rho_reference=rho,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    assert not np.any(isolated)

    positivity_floor = _mass_backed_gas_topology(
        np.array([0.20, 1.0e-4, 0.20]),
        rho * np.array([0.20, 1.0e-4, 0.20]),
        full_area=1.0,
        cell_width=1.0,
        rho_reference=rho,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.50,
    )
    np.testing.assert_array_equal(positivity_floor, [True, False, True])


def _uniform_state(params, nh=48, nv=32, dx=0.02, dz=0.02):
    ah = params.horizontal_area
    av = params.vertical_area
    alh = np.full(nh, 0.55 * ah)
    alv = np.full(nv, 0.55 * av)
    agh = ah - alh
    agv = av - alv
    hm = params.rho_atmospheric * agh * dx
    vm = params.rho_atmospheric * agv * dz
    return (
        hm,
        np.zeros(nh),
        vm,
        np.zeros(nv),
        np.zeros(nv),
        alh,
        np.zeros(nh),
        alv,
        np.zeros(nv),
    )


def test_taylor_bubble_mouth_retains_a_liquid_film() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )


def test_horizontal_front_threshold_follows_capillary_length() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
    assert 0.005 < params.horizontal_capillary_void_fraction < 0.012
    assert math.isclose(
        junction_mouth_area(1.0, params),
        params.vertical_gas_core_area_fraction * params.vertical_area,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_uniform_atmospheric_network_remains_at_rest() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = _uniform_state(params)
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=2.0e-4,
        junction_index=23,
        params=params,
    )
    np.testing.assert_allclose(result.horizontal_mass, state[0], atol=2.0e-13)
    np.testing.assert_allclose(result.vertical_total_mass, state[2], atol=2.0e-13)
    np.testing.assert_allclose(result.horizontal_momentum, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(result.vertical_momentum, 0.0, atol=2.0e-12)
    assert abs(result.junction_mass_transfer) < 2.0e-13
    assert abs(result.tracer_mass_error) < 2.0e-13


def test_vertical_gas_receives_shared_hydrostatic_buoyancy() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=9.81,
    )
    state = list(_uniform_state(params))
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=2.0e-5,
        junction_index=23,
        params=params,
    )
    assert float(np.mean(result.vertical_momentum[:-1])) > 0.0


def test_implicit_interphase_drag_conserves_momentum_without_overshoot() -> None:
    gas_mass = np.asarray([2.0e-4])
    gas_momentum = gas_mass * 40.0
    liquid_area = np.asarray([5.0e-5])
    liquid_discharge = liquid_area * -20.0
    gas_area = np.asarray([2.0e-3])
    interface = np.asarray([0.04])
    hydraulic = np.asarray([0.02])
    width = 0.04
    liquid_mass = 1000.0 * liquid_area * width
    momentum_before = float(
        gas_momentum[0]
        + liquid_mass[0] * liquid_discharge[0] / liquid_area[0]
    )
    relative_before = float(
        gas_momentum[0] / gas_mass[0]
        - liquid_discharge[0] / liquid_area[0]
    )

    gas_after, liquid_after = _implicit_interphase_drag_exchange(
        gas_mass,
        gas_momentum,
        liquid_area,
        liquid_discharge,
        gas_area,
        interface,
        hydraulic,
        cell_width=width,
        dt=0.02,
        rho_l=1000.0,
        gas_viscosity=1.81e-5,
    )
    momentum_after = float(
        gas_after[0]
        + liquid_mass[0] * liquid_after[0] / liquid_area[0]
    )
    relative_after = float(
        gas_after[0] / gas_mass[0]
        - liquid_after[0] / liquid_area[0]
    )
    assert math.isclose(
        momentum_after, momentum_before, rel_tol=0.0, abs_tol=1.0e-14
    )
    assert 0.0 < relative_after < relative_before


def test_holdup_enhancement_strengthens_only_internal_drag() -> None:
    gas_mass = np.asarray([2.0e-4])
    gas_momentum = gas_mass * 8.0
    liquid_area = np.asarray([1.5e-3])
    liquid_discharge = liquid_area * -0.5
    gas_area = np.asarray([5.0e-4])
    interface = np.asarray([0.04])
    hydraulic = np.asarray([0.02])
    width = 0.04
    liquid_mass = 1000.0 * liquid_area * width

    gas_base, liquid_base = _implicit_interphase_drag_exchange(
        gas_mass,
        gas_momentum,
        liquid_area,
        liquid_discharge,
        gas_area,
        interface,
        hydraulic,
        cell_width=width,
        dt=0.02,
        rho_l=1000.0,
        gas_viscosity=1.81e-5,
    )
    gas_enhanced, liquid_enhanced = _implicit_interphase_drag_exchange(
        gas_mass,
        gas_momentum,
        liquid_area,
        liquid_discharge,
        gas_area,
        interface,
        hydraulic,
        cell_width=width,
        dt=0.02,
        rho_l=1000.0,
        gas_viscosity=1.81e-5,
        liquid_holdup_drag_enhancement=75.0,
    )

    def total_momentum(gas, liquid):
        return float(gas[0] + liquid_mass[0] * liquid[0] / liquid_area[0])

    initial = total_momentum(gas_momentum, liquid_discharge)
    assert math.isclose(total_momentum(gas_base, liquid_base), initial, abs_tol=1e-14)
    assert math.isclose(
        total_momentum(gas_enhanced, liquid_enhanced), initial, abs_tol=1e-14
    )
    relative_base = gas_base[0] / gas_mass[0] - liquid_base[0] / liquid_area[0]
    relative_enhanced = (
        gas_enhanced[0] / gas_mass[0]
        - liquid_enhanced[0] / liquid_area[0]
    )
    assert 0.0 < relative_enhanced < relative_base


def test_confined_taylor_kinematics_does_not_double_apply_vertical_drag() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
        vertical_confined_interface_kinematics=True,
    )
    state = list(_uniform_state(params))
    state[3] = state[2] * 2.0
    state[4] = 0.5 * state[2]
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=2.0e-5,
        junction_index=23,
        params=params,
        vertical_pocket_front_height=0.64,
    )
    np.testing.assert_allclose(
        result.vertical_liquid_momentum_increment,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def test_pressure_driven_t_flux_is_conservative_and_horizontal_continues() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params))
    junction = 23
    state[0][: junction + 1] *= 1.015
    east_before = float(np.sum(state[0][junction + 1 :]))
    tracer_before = float(np.sum(state[0]) + np.sum(state[4]))
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=4.0e-4,
        junction_index=junction,
        params=params,
    )
    tracer_after = float(
        np.sum(result.horizontal_mass)
        + np.sum(result.vertical_tracer_mass)
        + result.escaped_tracer_mass
    )
    assert result.junction_mass_transfer > 0.0
    assert float(np.sum(result.vertical_tracer_mass)) > 0.0
    assert float(np.sum(result.horizontal_mass[junction + 1 :])) > east_before
    assert math.isclose(tracer_after, tracer_before, rel_tol=0.0, abs_tol=2.0e-11)
    assert abs(result.tracer_mass_error) < 2.0e-11


def test_resolved_vertical_receiving_hint_blocks_new_east_dead_leg_front() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    nh = 8
    nv = 6
    dx = dz = 0.02
    junction = 3
    ah = params.horizontal_area
    av = params.vertical_area
    h_al = np.full(nh, ah)
    h_al[: junction + 1] = 0.55 * ah
    # The east cell has an elastic/capillary-open area deficit but contains no
    # material gas.  It must remain closed while the riser characteristic says
    # that the vertical branch is receiving.
    h_al[junction + 1] = 0.95 * ah
    h_void = np.maximum(ah - h_al, params.void_floor_fraction * ah)
    h_mass = np.zeros(nh)
    h_mass[: junction + 1] = params.rho_atmospheric * h_void[: junction + 1] * dx
    h_momentum = np.zeros(nh)
    v_al = np.full(nv, av)
    v_void = np.maximum(av - v_al, params.void_floor_fraction * av)
    v_mass = params.rho_atmospheric * v_void * dz
    v_momentum = np.zeros(nv)
    v_tracer = np.zeros(nv)

    result = advance_coupled_gas_network(
        h_mass,
        h_momentum,
        v_mass,
        v_momentum,
        v_tracer,
        h_al,
        np.full(nh, 0.02 * ah),
        v_al,
        np.zeros(nv),
        dx=dx,
        dz=dz,
        dt=2.0e-4,
        junction_index=junction,
        params=params,
        vertical_branch_receiving_hint=True,
        horizontal_downstream_front_position=(junction + 1) * dx,
    )

    assert result.horizontal_mass[junction + 1] == 0.0
    assert math.isclose(
        result.downstream_front_position,
        (junction + 1) * dx,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert abs(result.total_mass_error) < 2.0e-13


def test_adjacent_sub_five_percent_crown_void_accepts_gas_front() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    dx = dz = 0.02
    nh, nv = 8, 6
    ah, av = params.horizontal_area, params.vertical_area
    h_liquid = np.full(nh, ah)
    h_liquid[:3] = 0.55 * ah
    h_liquid[3] = 0.99 * ah
    h_void = ah - h_liquid
    h_mass = np.zeros(nh)
    h_mass[:3] = params.rho_atmospheric * h_void[:3] * dx
    h_momentum = h_mass.copy()
    v_liquid = np.full(nv, av)
    v_mass = np.zeros(nv)

    result = advance_coupled_gas_network(
        h_mass,
        h_momentum,
        v_mass,
        np.zeros(nv),
        np.zeros(nv),
        h_liquid,
        np.zeros(nh),
        v_liquid,
        np.zeros(nv),
        dx=dx,
        dz=dz,
        dt=2.0e-4,
        junction_index=6,
        params=params,
    )

    assert result.horizontal_mass[3] > 0.0
    assert math.isclose(
        float(np.sum(result.horizontal_mass)
              + np.sum(result.vertical_tracer_mass)
              + result.escaped_tracer_mass),
        float(np.sum(h_mass)),
        rel_tol=0.0,
        abs_tol=2.0e-11,
    )


def test_vertical_tracer_cannot_outrun_the_fitted_material_front() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params))
    junction = 23
    state[0][: junction + 1] *= 1.015
    state[7][:] = params.vertical_area
    state[7][:3] = 0.55 * params.vertical_area
    state[7][-4:] = 0.0
    vertical_void = params.vertical_area - state[7]
    state[2] = params.rho_atmospheric * vertical_void * 0.02
    state[3][:] = 0.0
    state[4][:] = 0.0
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=4.0e-4,
        junction_index=junction,
        params=params,
        vertical_pocket_front_height=0.025,
    )
    # The newly exposed bottom cut cell is first filled by the conservative
    # acoustic remap, so the subsequent resolved Riemann flux may have either
    # sign.  Tunnel-origin gas must nevertheless enter the allowed front
    # domain and must not jump beyond it.
    assert float(np.sum(result.vertical_tracer_mass[:3])) > 0.0
    assert np.all(result.vertical_tracer_mass[3:] == 0.0)


def test_open_top_escape_uses_flux_and_closes_tracer_ledger() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params))
    # Close the T geometrically and put a pocket-gas tracer pulse near the open top.
    state[5][:] = params.horizontal_area
    state[0][:] = 0.0
    state[4][-4:] = 0.65 * state[2][-4:]
    state[3][-4:] = state[2][-4:] * 2.0
    tracer_before = float(np.sum(state[4]))
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=8.0e-4,
        junction_index=23,
        params=params,
    )
    assert result.escaped_tracer_mass > 0.0
    assert math.isclose(
        float(np.sum(result.vertical_tracer_mass)) + result.escaped_tracer_mass,
        tracer_before,
        rel_tol=0.0,
        abs_tol=2.0e-11,
    )


def test_unresolved_vacuum_tail_does_not_set_the_acoustic_time_step() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params))
    tail = 8
    state[0][tail] *= 0.10
    state[1][tail] = state[0][tail] * 5000.0
    total_before = float(np.sum(state[0]) + np.sum(state[2]))
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=1.0e-6,
        junction_index=23,
        params=params,
    )
    total_after = float(
        np.sum(result.horizontal_mass)
        + np.sum(result.vertical_total_mass)
        + result.atmospheric_mass_exchange
    )
    assert result.maximum_velocity < 1.0
    assert abs(result.horizontal_momentum[tail]) < 1.0e-18
    assert math.isclose(total_after, total_before, rel_tol=0.0, abs_tol=2.0e-11)


def test_gas_mouth_waits_for_resolved_riser_void() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params))
    junction = 23
    state[0][: junction + 1] *= 1.02
    state[7][:] = params.vertical_area
    floor_area = params.void_floor_fraction * params.vertical_area
    state[2][:] = params.rho_atmospheric * floor_area * 0.02
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=0.02,
        dt=2.0e-4,
        junction_index=junction,
        params=params,
    )
    assert result.junction_mouth_area == 0.0
    assert result.junction_mass_transfer == 0.0
    assert result.maximum_velocity < 10.0
    assert result.substeps < 50


def test_disconnected_massless_horizontal_void_is_not_a_gas_shortcut() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params, nh=30, nv=20))
    ah = params.horizontal_area
    av = params.vertical_area
    dx = 0.02
    dz = 0.02
    # True gas occupies cells 0:5.  A disconnected elastic rarefaction at
    # 20:24 has geometric void but no gas mass.
    state[5][:] = ah
    state[5][0:5] = 0.55 * ah
    state[5][20:24] = 0.80 * ah
    state[0][:] = 0.0
    state[0][0:5] = (
        params.rho_atmospheric * (ah - state[5][0:5]) * dx
    )
    state[7][:] = av
    state[2][:] = (
        params.rho_atmospheric
        * params.void_floor_fraction
        * av
        * dz
    )
    result = advance_coupled_gas_network(
        *state,
        dx=dx,
        dz=dz,
        dt=2.0e-4,
        junction_index=14,
        params=params,
    )
    np.testing.assert_allclose(result.horizontal_mass[20:24], 0.0, atol=1.0e-18)


def test_rarefied_gas_corridor_remains_a_connected_pressure_path() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params, nh=30, nv=20))
    dx = 0.02
    junction = 20
    void = params.horizontal_area - state[5]
    # Three consecutive rarefied cells separate two resolved gas regions.  The
    # centre cell is not a one-cell front receiver, so the old 50%-atmosphere
    # topology test made it an artificial wall despite its finite gas mass.
    state[0][2:5] = (
        0.10 * params.rho_atmospheric * void[2:5] * dx
    )
    centre_before = float(state[0][3])
    result = advance_coupled_gas_network(
        *state,
        dx=dx,
        dz=0.02,
        dt=2.0e-4,
        junction_index=junction,
        params=params,
    )
    assert not math.isclose(
        float(result.horizontal_mass[3]),
        centre_before,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    )


def test_inactive_top_cell_has_no_phantom_atmospheric_flux() -> None:
    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
        gravity=0.0,
    )
    state = list(_uniform_state(params, nh=30, nv=30))
    av = params.vertical_area
    dz = 0.02
    # A 3%-void top cell lies below the 5% atmospheric-connectivity threshold,
    # while the fitted tunnel-gas front is far below it.  Its stored mass must
    # not be exposed through a numerical floor-area boundary face.
    state[7][:] = av
    state[7][-1] = 0.97 * av
    void = np.maximum(av - state[7], params.void_floor_fraction * av)
    state[2] = params.rho_atmospheric * void * dz
    state[3][:] = 0.0
    state[4][:] = 0.0
    state[4][-1] = 0.05 * state[2][-1]
    total_before = float(np.sum(state[0]) + np.sum(state[2]))
    result = advance_coupled_gas_network(
        *state,
        dx=0.02,
        dz=dz,
        dt=2.0e-4,
        junction_index=20,
        params=params,
        vertical_pocket_front_height=0.30,
    )
    total_after = float(
        np.sum(result.horizontal_mass)
        + np.sum(result.vertical_total_mass)
        + result.atmospheric_mass_exchange
    )
    assert result.atmospheric_mass_exchange == 0.0
    assert math.isclose(total_after, total_before, rel_tol=0.0, abs_tol=2.0e-13)
    assert abs(result.total_mass_error) < 2.0e-13
