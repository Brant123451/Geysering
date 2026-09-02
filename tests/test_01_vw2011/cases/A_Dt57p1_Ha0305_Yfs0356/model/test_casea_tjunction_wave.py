from __future__ import annotations

import math

import numpy as np

from vw2011_network_twofluid import (
    _advance_horizontal_liquid_hyperbolic_ssprk2,
    _branch_consistent_external_pressure_gradient,
    _bulk_material_reaches_riser_outlet,
    _advance_riser_taylor_front,
    _apply_finite_width_side_t_exchange,
    _apply_finite_width_side_t_gross_exchange,
    _connected_pocket_inventory,
    _connected_shallow_water_potential_offsets,
    _collapse_upper_slug_at_taylor_breakthrough,
    _remap_vertical_gas_for_liquid_relocation,
    _countercurrent_flooding_liquid_flow,
    _decoupled_liquid_rusanov_flux,
    _decoupled_restoring_coefficient,
    _displace_newly_swept_taylor_slice,
    _fit_riser_taylor_core,
    _implicit_smagorinsky_momentum_diffusion,
    _limit_gas_void_closure_flux,
    _limit_liquid_donor_flux,
    _limit_three_branch_junction_flows,
    _mass_supported_vertical_gas_mouth,
    _orthogonal_junction_liquid_exchange,
    _open_riser_annular_film_flux,
    _open_side_t_east_capillary_cutcell,
    _project_riser_taylor_topology,
    _project_single_liquid_column,
    _regularize_near_dry_momentum,
    _relax_elastic_riser_storage_for_twostream_handoff,
    _return_refilled_taylor_core_to_side_t,
    _restore_refilled_taylor_core_to_unswept_slug,
    _riser_acoustic_momentum_dissipation_flux,
    _return_new_taylor_sweep_to_side_t,
    _return_isolated_top_bulk_liquid,
    _riser_liquid_friction_rate,
    _riser_material_gas_mask,
    _top_connected_atmospheric_gas_mask,
    _equilibrate_open_riser_headspace,
    _replace_horizontal_face_with_tjunction_fluxes,
    _restore_riser_annular_film,
    _section_hydrostatic,
    _sharpen_unswept_riser_liquid_slug,
    _two_phase_mixing_activation,
    _side_t_opening_weights,
    _side_t_return_exchange_weights,
    _limit_side_t_downward_liquid_flow,
    _limit_side_t_upward_liquid_flow,
    _limit_riser_bottom_inflow_by_receiving_capacity,
    _limit_taylor_return_exchange_flow,
    _vw_laminar_film_closure,
    _vertical_liquid_holdup_head,
    _vertical_two_phase_mouth_pressure,
)
from casea_horizontal_liquid_operator import (
    HorizontalLiquidParameters,
    pressure_potential_state,
)
from casea_vertical_twostream_fv import (
    VerticalTwoStreamParameters,
    VerticalTwoStreamState,
    advance_vertical_two_stream_fv,
)


def test_connected_potential_offsets_equal_endpoint_traction_matches() -> None:
    params = HorizontalLiquidParameters(
        area_full=math.pi * 0.094**2 / 4.0,
        diameter=0.094,
        wave_speed=28.0,
        cell_width=0.08,
    )
    area = params.area_full * np.array([1.0, 0.55, 0.60, 1.0, 0.48, 1.0])
    discharge = np.array([0.0, 1.0e-4, 1.5e-4, 0.0, -0.5e-4, 0.0])
    mass = np.full(6, 1.0e-5)
    momentum = np.zeros(6)
    support = np.array([False, True, True, False, True, False])

    offsets = _connected_shallow_water_potential_offsets(
        area,
        support,
        params,
    )
    natural = pressure_potential_state(
        area,
        discharge,
        mass,
        momentum,
        support,
        params,
    ).potential
    elastic = pressure_potential_state(
        area,
        discharge,
        np.zeros(6),
        np.zeros(6),
        np.zeros(6, dtype=bool),
        params,
    ).potential
    expected_first = math.fsum(
        (elastic[0] - natural[1], elastic[3] - natural[2])
    ) / 2.0
    expected_second = math.fsum(
        (elastic[3] - natural[4], elastic[5] - natural[4])
    ) / 2.0
    np.testing.assert_allclose(offsets, [0.0, expected_first, expected_first, 0.0, expected_second, 0.0])


def test_riser_acoustic_momentum_flux_damps_odd_even_mode_conservatively() -> None:
    discharge = np.array([1.0, -1.0, 1.0, -1.0])
    celerity = np.full(4, 2.0)
    flux = _riser_acoustic_momentum_dissipation_flux(
        discharge,
        celerity,
        np.ones(4),
        full_area=1.0,
    )
    np.testing.assert_allclose(flux, [0.0, 2.0, -2.0, 2.0, 0.0])

    updated = discharge - 0.10 * (flux[1:] - flux[:-1])
    assert float(np.dot(updated, updated)) < float(
        np.dot(discharge, discharge)
    )
    assert math.isclose(float(np.sum(updated)), float(np.sum(discharge)))
    np.testing.assert_array_equal(
        _riser_acoustic_momentum_dissipation_flux(
            np.full(4, 0.25),
            celerity,
            np.ones(4),
            full_area=1.0,
        ),
        np.zeros(5),
    )
    np.testing.assert_array_equal(
        _riser_acoustic_momentum_dissipation_flux(
            discharge,
            celerity,
            np.array([1.0, 0.01, 1.0, 1.0]),
            full_area=1.0,
        ),
        np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
    )


def test_external_pressure_source_does_not_cross_tpa_material_face() -> None:
    """The mixed-face traction belongs to the conservative Riemann flux."""

    pressure = np.array([101400.0, 101500.0, 95000.0, 94920.0])
    stratified = np.array([True, True, False, False])
    gradient = _branch_consistent_external_pressure_gradient(
        pressure,
        stratified,
        cell_width=0.10,
    )

    # The 6.5-kPa branch datum jump is absent.  The physical residual
    # gradients within the gas-supported and elastic branches remain.
    np.testing.assert_allclose(
        gradient,
        np.array([500.0, 500.0, -400.0, -400.0]),
    )


def test_uniform_residual_pressure_has_zero_branch_consistent_gradient() -> None:
    pressure = np.full(6, 101325.0)
    stratified = np.array([True, True, True, False, False, False])
    np.testing.assert_array_equal(
        _branch_consistent_external_pressure_gradient(
            pressure,
            stratified,
            cell_width=0.02,
        ),
        np.zeros(6),
    )


def test_gross_side_t_exchange_does_not_cancel_countercurrent_parcels() -> None:
    area = np.array([0.6, 0.8, 0.4])
    discharge = np.array([0.12, -0.08, 0.02])
    weights = np.array([0.25, 0.50, 0.25])
    dt = 0.01
    dx = 0.10
    q_up = 0.20
    q_down = 0.15
    before = float(np.sum(area) * dx)
    updated_area, updated_q = _apply_finite_width_side_t_gross_exchange(
        area,
        discharge,
        upward_flow=q_up,
        downward_flow=q_down,
        opening_weights=weights,
        dt=dt,
        cell_width=dx,
        full_area=1.0,
    )

    assert math.isclose(
        float(np.sum(updated_area) * dx) - before,
        (q_down - q_up) * dt,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    # Upward withdrawal carries away donor momentum; the orthogonal return
    # adds volume but no prescribed horizontal impulse.
    assert np.linalg.norm(updated_q) < np.linalg.norm(discharge)


def test_side_t_return_uses_liquid_continuous_part_of_finite_mouth() -> None:
    area = np.array([0.60, 1.00, 0.20])
    opening = np.array([0.65, 0.35, 0.00])
    gas_supported = np.array([True, False, False])
    return_weights = _side_t_return_exchange_weights(
        area,
        opening,
        full_area=1.0,
        gas_supported=gas_supported,
    )
    np.testing.assert_array_equal(return_weights, [0.0, 1.0, 0.0])

    updated_area, _ = _apply_finite_width_side_t_gross_exchange(
        area,
        np.zeros_like(area),
        upward_flow=0.0,
        downward_flow=0.10,
        opening_weights=opening,
        downward_weights=return_weights,
        dt=0.01,
        cell_width=0.10,
        full_area=1.0,
    )
    # The lower liquid path accepts the orthogonal return as elastic storage;
    # the gas-crown control volume is not numerically crushed.
    assert updated_area[0] == area[0]
    assert updated_area[1] > area[1]


def test_riser_bottom_inflow_uses_void_plus_simultaneous_throughflow() -> None:
    area = np.array([0.90, 0.70])
    face_flux = np.array([0.0, 0.03, 0.0])
    accepted = _limit_riser_bottom_inflow_by_receiving_capacity(
        area,
        face_flux,
        requested_flow=0.20,
        dt=0.10,
        cell_width=0.10,
        full_area=1.0,
    )
    # 0.10*0.10/0.10 = 0.10 storage plus 0.03 through-flow.
    assert math.isclose(accepted, 0.13)
    updated_bottom = area[0] - 0.10 / 0.10 * (face_flux[1] - accepted)
    assert math.isclose(updated_bottom, 1.0)


def test_riser_bottom_inflow_closes_when_full_cell_has_downward_face_flux() -> None:
    accepted = _limit_riser_bottom_inflow_by_receiving_capacity(
        np.array([1.0, 0.8]),
        np.array([0.0, -0.02, 0.0]),
        requested_flow=0.10,
        dt=0.05,
        cell_width=0.10,
        full_area=1.0,
    )
    assert accepted == 0.0


def test_open_riser_headspace_stops_at_liquid_seal_or_material_gas() -> None:
    gas_area = np.array([0.8, 0.0, 0.2, 0.8, 1.0])
    tracer = np.array([0.0, 0.0, 0.03, 0.0, 0.0])
    mask = _top_connected_atmospheric_gas_mask(
        gas_area,
        tracer,
        cell_length=0.1,
        reference_density=1.0,
        dry_area_tolerance=1.0e-12,
    )
    np.testing.assert_array_equal(mask, [False, False, False, True, True])


def test_riser_eos_rejects_total_mass_in_untraced_microvoid() -> None:
    full_area = 1.0
    gas_area = np.array([1.0e-5, 0.8, 0.8])
    tracer = np.array([0.0, 0.08, 0.0])
    mask = _riser_material_gas_mask(
        gas_area,
        tracer,
        full_area=full_area,
        cell_length=0.1,
        reference_density=1.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        resolved_density_fraction=0.5,
    )
    np.testing.assert_array_equal(mask, [False, True, False])


def test_open_riser_headspace_projects_only_ambient_mass_and_books_exchange() -> None:
    mass = np.array([0.08, 0.02, 0.20, 0.30])
    momentum = np.array([0.04, -0.01, 0.10, -0.15])
    tracer = np.array([0.04, 0.0, 0.0, 0.0])
    gas_area = np.array([0.8, 0.0, 0.5, 1.0])
    projected_mass, projected_momentum, exchange, mask = (
        _equilibrate_open_riser_headspace(
            mass,
            momentum,
            tracer,
            gas_area,
            cell_length=0.1,
            reference_density=1.0,
            dry_area_tolerance=1.0e-12,
        )
    )

    np.testing.assert_array_equal(mask, [False, False, True, True])
    np.testing.assert_allclose(projected_mass, [0.08, 0.02, 0.05, 0.10])
    # Removing ambient mass carries its local momentum at unchanged velocity.
    np.testing.assert_allclose(
        projected_momentum,
        [0.04, -0.01, 0.025, -0.05],
    )
    assert math.isclose(exchange, 0.35)
    assert math.isclose(
        float(np.sum(projected_mass)) + exchange,
        float(np.sum(mass)),
    )


def test_new_top_connected_taylor_void_receives_atmospheric_mass() -> None:
    mass = np.zeros(3)
    momentum = np.zeros(3)
    tracer = np.zeros(3)
    gas_area = np.array([0.25, 0.50, 1.00])

    projected_mass, projected_momentum, exchange, mask = (
        _equilibrate_open_riser_headspace(
            mass,
            momentum,
            tracer,
            gas_area,
            cell_length=0.10,
            reference_density=1.20,
            dry_area_tolerance=1.0e-12,
        )
    )

    np.testing.assert_array_equal(mask, [True, True, True])
    np.testing.assert_allclose(projected_mass, [0.03, 0.06, 0.12])
    np.testing.assert_allclose(projected_momentum, 0.0)
    assert math.isclose(exchange, -0.21)


def test_riser_liquid_outlet_requires_bulk_material_to_reach_lip() -> None:
    full = 2.0
    liquid = np.array([full, full, full, 0.5 * full, 0.0, 0.0])
    material_gas = np.zeros_like(liquid)

    assert not _bulk_material_reaches_riser_outlet(
        liquid,
        material_gas,
        full_area=full,
        cell_width=0.10,
        riser_height=0.60,
    )

    material_gas[3:] = np.array([0.5, 1.0, 1.0])
    assert _bulk_material_reaches_riser_outlet(
        liquid,
        material_gas,
        full_area=full,
        cell_width=0.10,
        riser_height=0.60,
    )


def test_isolated_top_liquid_remnant_does_not_open_riser_outlet() -> None:
    liquid = np.array([1.0, 1.0, 1.0, 0.4, 0.0, 1.0e-3])
    material_gas = np.zeros_like(liquid)

    assert not _bulk_material_reaches_riser_outlet(
        liquid,
        material_gas,
        full_area=1.0,
        cell_width=0.10,
        riser_height=0.60,
    )


def test_unswept_slug_sharpening_is_local_and_conservative() -> None:
    area = np.array([0.25, 0.35, 1.0, 0.45, 0.05, 0.10])
    discharge = np.array([0.01, -0.02, 0.30, 0.18, -0.01, 0.02])
    sharpened_area, sharpened_q = _sharpen_unswept_riser_liquid_slug(
        area,
        discharge,
        material_front_height=0.20,
        full_area=1.0,
        dz=0.10,
    )

    # Cells swept by the nose are not part of the topology closure.
    np.testing.assert_array_equal(sharpened_area[:2], area[:2])
    np.testing.assert_array_equal(sharpened_q[:2], discharge[:2])
    np.testing.assert_allclose(sharpened_area[2:], [1.0, 0.60, 0.0, 0.0])
    np.testing.assert_allclose(
        np.sum(sharpened_area[2:]),
        np.sum(area[2:]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(sharpened_q[2:]),
        np.sum(discharge[2:]),
        rtol=0.0,
        atol=1.0e-14,
    )

    second_area, second_q = _sharpen_unswept_riser_liquid_slug(
        sharpened_area,
        sharpened_q,
        material_front_height=0.20,
        full_area=1.0,
        dz=0.10,
    )
    np.testing.assert_allclose(second_area, sharpened_area)
    np.testing.assert_allclose(second_q, sharpened_q)

    resting_area, resting_q = _sharpen_unswept_riser_liquid_slug(
        area,
        discharge,
        material_front_height=0.20,
        full_area=1.0,
        dz=0.10,
        slug_velocity=0.0,
    )
    np.testing.assert_allclose(resting_area, sharpened_area)
    np.testing.assert_array_equal(resting_q[:2], discharge[:2])
    np.testing.assert_array_equal(resting_q[2:], 0.0)


def test_breakthrough_collapse_returns_upper_slug_to_geometric_mixing_zone() -> None:
    area = np.array([0.20, 0.30, 0.20, 0.90, 0.80, 0.40])
    discharge = np.array([0.01, -0.02, 0.03, -0.04, 0.50, 0.60])
    collapsed_area, collapsed_q, returned = (
        _collapse_upper_slug_at_taylor_breakthrough(
            area,
            discharge,
            material_front_height=0.40,
            full_area=1.0,
            dz=0.10,
            mixing_zone_height=0.20,
        )
    )

    np.testing.assert_allclose(returned, 0.12)
    np.testing.assert_allclose(collapsed_area, [1.0, 0.70, 0.20, 0.90, 0.0, 0.0])
    np.testing.assert_allclose(np.sum(collapsed_area), np.sum(area))
    np.testing.assert_array_equal(collapsed_q[:4], discharge[:4])
    np.testing.assert_array_equal(collapsed_q[4:], 0.0)


def test_liquid_relocation_remaps_gas_parcels_without_changing_state() -> None:
    full_area = 1.0
    dz = 0.25
    old_liquid = np.array([0.20, 0.40, 0.80, 0.60])
    new_liquid = np.array([0.80, 0.60, 0.40, 0.20])

    old_void = (full_area - old_liquid) * dz
    new_void = (full_area - new_liquid) * dz
    density = np.array([1.20, 0.90, 1.10, 1.00])
    tracer_fraction = np.array([0.25, 0.60, 0.40, 0.10])
    velocity = np.array([-2.00, 3.00, 0.50, -0.25])
    gas_mass = density * old_void
    tracer_mass = tracer_fraction * gas_mass
    gas_momentum = velocity * gas_mass

    opened_void = np.maximum(old_liquid - new_liquid, 0.0) * dz
    closed_void = np.maximum(new_liquid - old_liquid, 0.0) * dz
    np.testing.assert_allclose(
        np.sum(old_liquid) * dz,
        np.sum(new_liquid) * dz,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        np.sum(opened_void),
        np.sum(closed_void),
        rtol=0.0,
        atol=1.0e-15,
    )

    remapped_mass, remapped_tracer, remapped_momentum = (
        _remap_vertical_gas_for_liquid_relocation(
            old_liquid,
            new_liquid,
            gas_mass,
            tracer_mass,
            gas_momentum,
            full_area=full_area,
            dz=dz,
        )
    )

    np.testing.assert_allclose(
        remapped_mass,
        np.array([0.060, 0.090, 0.175, 0.205]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        remapped_tracer,
        np.array([0.015, 0.054, 0.052, 0.052]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        remapped_momentum,
        np.array([-0.120, 0.270, -0.2125, -0.010]),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(remapped_mass),
        np.sum(gas_mass),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(remapped_tracer),
        np.sum(tracer_mass),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(remapped_momentum),
        np.sum(gas_momentum),
        rtol=0.0,
        atol=1.0e-14,
    )

    closing_cells = new_liquid > old_liquid
    np.testing.assert_allclose(
        remapped_mass[closing_cells] / new_void[closing_cells],
        gas_mass[closing_cells] / old_void[closing_cells],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        remapped_tracer[closing_cells] / remapped_mass[closing_cells],
        tracer_mass[closing_cells] / gas_mass[closing_cells],
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        remapped_momentum[closing_cells] / remapped_mass[closing_cells],
        gas_momentum[closing_cells] / gas_mass[closing_cells],
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_liquid_relocation_rejects_mismatched_volume() -> None:
    with np.testing.assert_raises_regex(
        FloatingPointError,
        "liquid relocation does not conserve volume",
    ):
        _remap_vertical_gas_for_liquid_relocation(
            np.array([0.20, 0.40]),
            np.array([0.30, 0.40]),
            np.array([0.08, 0.06]),
            np.array([0.02, 0.03]),
            np.array([-0.04, 0.12]),
            full_area=1.0,
            dz=0.10,
        )


def test_postbreakthrough_top_island_returns_only_when_disconnected() -> None:
    area = np.array([0.20, 0.30, 0.20, 0.05, 0.02, 0.90])
    discharge = np.array([0.01, -0.02, 0.03, 0.04, 0.05, 2.0])
    returned_area, returned_q, returned = _return_isolated_top_bulk_liquid(
        area,
        discharge,
        material_front_height=0.30,
        full_area=1.0,
        dz=0.10,
        mixing_zone_height=0.20,
    )
    np.testing.assert_allclose(returned, 0.09)
    np.testing.assert_allclose(returned_area, [1.0, 0.40, 0.20, 0.05, 0.02, 0.0])
    np.testing.assert_allclose(np.sum(returned_area), np.sum(area))
    np.testing.assert_array_equal(returned_q[:5], discharge[:5])
    assert returned_q[5] == 0.0

    connected = np.array([0.20, 0.30, 0.20, 0.60, 0.70, 0.90])
    unchanged_area, unchanged_q, unchanged = _return_isolated_top_bulk_liquid(
        connected,
        discharge,
        material_front_height=0.30,
        full_area=1.0,
        dz=0.10,
        mixing_zone_height=0.20,
    )
    assert unchanged == 0.0
    np.testing.assert_array_equal(unchanged_area, connected)
    np.testing.assert_array_equal(unchanged_q, discharge)


def test_vertical_two_phase_mouth_uses_connected_interface_pressure() -> None:
    liquid_trace = 104_494.31
    connected_gas = 108_972.17
    full_area = math.pi * 0.0571**2 / 4.0

    assert _vertical_two_phase_mouth_pressure(
        liquid_trace_pressure=liquid_trace,
        connected_gas_pressure=connected_gas,
        gas_mouth_area=0.0,
        full_area=full_area,
    ) == liquid_trace
    assert _vertical_two_phase_mouth_pressure(
        liquid_trace_pressure=liquid_trace,
        connected_gas_pressure=connected_gas,
        gas_mouth_area=0.2 * full_area,
        full_area=full_area,
    ) == connected_gas


def test_shared_two_phase_mouth_pressure_has_no_false_fv_acceleration() -> None:
    """The node endpoint and FV start face form one hydrostatic pressure line."""

    parameters = VerticalTwoStreamParameters(
        cell_count=1,
        cell_length=0.16,
        diameter=0.0571,
        liquid_density=998.0,
        gravity=9.81,
        wall_friction_up=0.0,
        wall_friction_down=0.0,
        interstream_drag=0.0,
    )
    mouth_pressure = _vertical_two_phase_mouth_pressure(
        liquid_trace_pressure=104_494.31,
        connected_gas_pressure=108_972.17,
        gas_mouth_area=0.2 * parameters.full_area,
        full_area=parameters.full_area,
    )
    state = VerticalTwoStreamState.from_iterables(
        upward_area=[0.6 * parameters.full_area],
        upward_discharge=[0.0],
        downward_area=[0.2 * parameters.full_area],
        downward_discharge=[0.0],
    )
    step = advance_vertical_two_stream_fv(
        state,
        parameters,
        dt=1.0e-3,
        pressure_faces=(
            mouth_pressure,
            mouth_pressure
            - parameters.liquid_density
            * parameters.gravity
            * parameters.cell_length,
        ),
    )

    np.testing.assert_allclose(step.state.upward_discharge, (0.0,), atol=1.0e-15)
    np.testing.assert_allclose(step.state.downward_discharge, (0.0,), atol=1.0e-15)
    assert abs(step.ledger.pressure_gravity_impulse) < 1.0e-15


def test_ccfl_leaves_upward_flow_and_limits_only_excess_downward_flow() -> None:
    kwargs = dict(
        upward_gas_superficial_velocity=0.02,
        full_area=2.5e-3,
        diameter=0.0571,
        rho_l=998.0,
        rho_g=1.2,
        gravity=9.81,
        wallis_constant=0.40,
    )
    upward = _countercurrent_flooding_liquid_flow(2.0e-4, **kwargs)
    modest_downward = _countercurrent_flooding_liquid_flow(-1.0e-4, **kwargs)
    excessive_downward = _countercurrent_flooding_liquid_flow(-1.0e-3, **kwargs)

    assert upward == 2.0e-4
    assert modest_downward == -1.0e-4
    assert -1.0e-3 < excessive_downward < -1.0e-4


def test_ccfl_upward_gas_reduces_downward_liquid_capacity() -> None:
    kwargs = dict(
        requested_flow=-1.0e-3,
        full_area=2.5e-3,
        diameter=0.0571,
        rho_l=998.0,
        rho_g=1.2,
        gravity=9.81,
        wallis_constant=0.40,
    )
    stagnant_gas = _countercurrent_flooding_liquid_flow(
        upward_gas_superficial_velocity=0.0,
        **kwargs,
    )
    weakly_rising_gas = _countercurrent_flooding_liquid_flow(
        upward_gas_superficial_velocity=0.02,
        **kwargs,
    )
    rising_gas = _countercurrent_flooding_liquid_flow(
        upward_gas_superficial_velocity=0.5,
        **kwargs,
    )
    assert stagnant_gas == kwargs["requested_flow"]
    assert weakly_rising_gas > stagnant_gas
    assert rising_gas > weakly_rising_gas


def test_open_annular_film_recovers_vw_terminal_flux_and_thins_smoothly() -> None:
    diameter = 0.0571
    thickness, core_fraction, terminal_flow, _ = _vw_laminar_film_closure(
        diameter
    )
    full = 0.25 * math.pi * diameter**2
    film_area = (1.0 - core_fraction) * full
    flow, velocity = _open_riser_annular_film_flux(
        film_area,
        diameter=diameter,
        maximum_flow=terminal_flow,
    )
    np.testing.assert_allclose(flow, -terminal_flow, rtol=2.0e-3)
    assert velocity < 0.0
    thin_flow, _ = _open_riser_annular_film_flux(
        0.5 * film_area,
        diameter=diameter,
        maximum_flow=terminal_flow,
    )
    assert 0.0 > thin_flow > flow
    zero_flow, zero_velocity = _open_riser_annular_film_flux(
        0.0,
        diameter=diameter,
        maximum_flow=terminal_flow,
    )
    assert zero_flow == 0.0
    assert zero_velocity == 0.0


def test_vertical_gas_mouth_requires_mass_support_and_keeps_open_core_area() -> None:
    full = 2.5e-3
    dz = 0.02
    rho = 1.2
    void = 0.90 * full
    liquid = full - void
    unsupported = _mass_supported_vertical_gas_mouth(
        liquid,
        0.01 * rho * void * dz,
        full_area=full,
        cell_width=dz,
        rho_reference=rho,
        density_fraction=0.02,
        maximum_gas_area_fraction=0.80,
    )
    assert unsupported == 0.0
    supported = _mass_supported_vertical_gas_mouth(
        liquid,
        rho * void * dz,
        full_area=full,
        cell_width=dz,
        rho_reference=rho,
        density_fraction=0.02,
        maximum_gas_area_fraction=0.80,
    )
    np.testing.assert_allclose(supported, 0.80 * full)
    closed = _mass_supported_vertical_gas_mouth(
        full,
        rho * full * dz,
        full_area=full,
        cell_width=dz,
        rho_reference=rho,
        density_fraction=0.02,
        maximum_gas_area_fraction=0.80,
    )
    assert closed == 0.0


def test_junction_pressure_inventory_excludes_disconnected_gas() -> None:
    full = 1.0
    dx = 0.1
    area = np.array([0.4, 0.5, 1.0, 1.0, 0.7, 0.6, 1.0])
    mass = np.array([1.0, 2.0, 9.0, 8.0, 4.0, 5.0, 7.0])
    mask = np.array([True, True, False, False, True, True, False])
    connected_mass, connected_volume = _connected_pocket_inventory(
        area,
        mass,
        mask,
        index=4,
        full_area=full,
        cell_width=dx,
    )
    # The component is cells 4:6; its one-cell mass halo is cells 3:7.
    assert connected_mass == 24.0
    np.testing.assert_allclose(connected_volume, 0.07)


def test_two_phase_mixing_loss_activates_continuously_with_holdup() -> None:
    assert _two_phase_mixing_activation(0.0) == 0.0
    assert _two_phase_mixing_activation(1.0) == 0.0
    assert _two_phase_mixing_activation(0.5) == 1.0
    np.testing.assert_allclose(
        _two_phase_mixing_activation(0.02),
        4.0 * 0.02 * 0.98,
    )


def test_finite_width_side_t_weights_follow_physical_opening() -> None:
    weights = _side_t_opening_weights(
        200,
        cell_width=0.02,
        junction_center=3.520,
        opening_width=0.0571,
    )
    np.testing.assert_allclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-15)
    assert np.count_nonzero(weights) == 4
    active = weights[weights > 0.0]
    np.testing.assert_allclose(active, active[::-1], rtol=0.0, atol=2.0e-14)


def test_finite_width_side_t_exchange_conserves_liquid_and_momentum_rule() -> None:
    area = np.full(10, 4.0e-3)
    discharge = area * np.linspace(0.1, 1.0, area.size)
    weights = _side_t_opening_weights(
        area.size,
        cell_width=0.04,
        junction_center=0.20,
        opening_width=0.0571,
    )
    returned_area, returned_discharge = _apply_finite_width_side_t_exchange(
        area,
        discharge,
        upward_flow=-8.0e-4,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.04,
    )
    np.testing.assert_allclose(
        np.sum(returned_area - area) * 0.04,
        8.0e-6,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_array_equal(returned_discharge, discharge)

    removed_area, removed_discharge = _apply_finite_width_side_t_exchange(
        area,
        discharge,
        upward_flow=8.0e-4,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.04,
    )
    active = weights > 0.0
    np.testing.assert_allclose(
        removed_discharge[active] / removed_area[active],
        discharge[active] / area[active],
    )


def test_side_t_return_adds_no_prescribed_axial_impulse() -> None:
    area = np.full(12, 4.0e-3)
    discharge = np.zeros_like(area)
    weights = _side_t_opening_weights(
        area.size,
        cell_width=0.02,
        junction_center=0.12,
        opening_width=0.0571,
    )
    area_after, discharge_after = _apply_finite_width_side_t_exchange(
        area,
        discharge,
        upward_flow=-6.0e-4,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.02,
    )
    np.testing.assert_array_equal(discharge_after, discharge)
    np.testing.assert_allclose(
        np.sum(area_after - area) * 0.02,
        6.0e-6,
        rtol=0.0,
        atol=1.0e-16,
    )


def test_side_t_return_uses_open_compliance_before_compressing_full_cell() -> None:
    full = 6.0e-3
    area = np.array([full, 0.25 * full, full, full])
    weights = np.array([0.0, 0.5, 0.5, 0.0])
    returned = 3.0e-6
    updated, discharge = _apply_finite_width_side_t_exchange(
        area,
        np.zeros_like(area),
        upward_flow=-returned / 0.01,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.02,
        full_area=full,
    )
    assert updated[2] == full
    assert updated[1] > area[1]
    np.testing.assert_allclose(
        np.sum(updated - area) * 0.02,
        returned,
        rtol=0.0,
        atol=1.0e-16,
    )
    np.testing.assert_array_equal(discharge, 0.0)


def test_finite_width_side_t_outflow_limit_preserves_all_donors() -> None:
    area = np.array([4.0e-3, 1.0e-4, 2.0e-3, 4.0e-3])
    weights = np.array([0.0, 0.25, 0.75, 0.0])
    limited = _limit_side_t_upward_liquid_flow(
        area,
        requested_flow=1.0,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.04,
        retained_fraction=0.10,
    )
    updated, _ = _apply_finite_width_side_t_exchange(
        area,
        np.zeros_like(area),
        upward_flow=limited,
        opening_weights=weights,
        dt=0.01,
        cell_width=0.04,
    )
    assert np.all(updated[weights > 0.0] >= 0.10 * area[weights > 0.0] - 1.0e-15)
    np.testing.assert_allclose(
        _two_phase_mixing_activation(0.02 - 1.0e-8),
        _two_phase_mixing_activation(0.02 + 1.0e-8),
        rtol=0.0,
        atol=8.0e-8,
    )


def test_side_t_return_limiter_retains_mass_supported_gas_void() -> None:
    full = 6.0e-3
    dx = 0.02
    dt = 0.01
    rho = 1.2
    weights = np.array([0.0, 0.25, 0.75, 0.0])
    area = np.array([full, 0.90 * full, 0.80 * full, full])
    gas_mass = rho * np.maximum(full - area, 1.0e-4 * full) * dx
    limited = _limit_side_t_downward_liquid_flow(
        area,
        gas_mass,
        requested_flow=-1.0,
        opening_weights=weights,
        dt=dt,
        cell_width=dx,
        full_area=full,
        rho_reference=rho,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
    )
    updated, _ = _apply_finite_width_side_t_exchange(
        area,
        np.zeros_like(area),
        upward_flow=limited,
        opening_weights=weights,
        dt=dt,
        cell_width=dx,
    )
    minimum_void = gas_mass / (2.0 * rho * dx)
    active = weights > 0.0
    assert np.all(full - updated[active] >= minimum_void[active] - 1.0e-15)


def test_side_t_return_ignores_liquid_full_massless_footprint_cell() -> None:
    full = 6.0e-3
    dx = 0.02
    dt = 0.01
    rho = 1.2
    weights = np.array([0.0, 0.25, 0.75, 0.0])
    area = np.array([full, full, 0.80 * full, full])
    gas_mass = rho * np.maximum(full - area, 1.0e-4 * full) * dx
    requested = -2.0e-4
    limited = _limit_side_t_downward_liquid_flow(
        area,
        gas_mass,
        requested_flow=requested,
        opening_weights=weights,
        dt=dt,
        cell_width=dx,
        full_area=full,
        rho_reference=rho,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
    )
    assert limited < 0.0
    assert limited >= requested


def test_taylor_return_override_reapplies_bottom_cell_donor_limit() -> None:
    dt = 0.01
    dz = 0.02
    dx = 0.02
    riser_area = np.array([4.0e-3, 3.0e-3, 2.0e-3])
    # The Taylor closure replaces face 0.  Face 1 already carries a distinct
    # upward outflow that must keep its share of the bottom-cell donor budget.
    riser_flux = np.array([4.0e-4, 2.0e-4, 0.0, 0.0])
    horizontal_full = 6.0e-3
    horizontal_area = np.full(4, horizontal_full)
    weights = np.array([0.0, 0.25, 0.75, 0.0])

    limited = _limit_taylor_return_exchange_flow(
        riser_area,
        riser_flux,
        horizontal_area,
        np.zeros_like(horizontal_area),
        requested_return_flow=1.0e-2,
        opening_weights=weights,
        dt=dt,
        riser_cell_width=dz,
        horizontal_cell_width=dx,
        horizontal_full_area=horizontal_full,
        rho_reference=1.2,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        retained_fraction=0.10,
    )

    expected = (
        0.90 * riser_area[0] * dz / dt - riser_flux[1]
    )
    np.testing.assert_allclose(limited, expected, rtol=0.0, atol=1.0e-15)
    overridden = riser_flux.copy()
    overridden[0] = -limited
    riser_after = riser_area - dt / dz * (
        overridden[1:] - overridden[:-1]
    )
    assert riser_after[0] >= 0.10 * riser_area[0] - 1.0e-15


def test_taylor_return_override_conserves_both_branches_and_is_local() -> None:
    dt = 0.01
    dz = 0.02
    dx = 0.02
    rho = 1.2
    riser_area = np.array([4.0e-3, 3.0e-3, 2.0e-3])
    old_flux = np.array([4.0e-4, 2.0e-4, 0.0, 0.0])
    horizontal_full = 6.0e-3
    horizontal_area = np.array(
        [horizontal_full, 3.0e-3, 4.0e-3, horizontal_full]
    )
    horizontal_discharge = np.array([0.1, -0.2, 0.3, -0.4]) * horizontal_area
    weights = np.array([0.0, 0.25, 0.75, 0.0])
    raw_void = np.maximum(horizontal_full - horizontal_area, 0.0)
    horizontal_gas_mass = rho * raw_void * dx

    limited = _limit_taylor_return_exchange_flow(
        riser_area,
        old_flux,
        horizontal_area,
        horizontal_gas_mass,
        requested_return_flow=1.0e-2,
        opening_weights=weights,
        dt=dt,
        riser_cell_width=dz,
        horizontal_cell_width=dx,
        horizontal_full_area=horizontal_full,
        rho_reference=rho,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        topology_density_fraction=0.02,
        retained_fraction=0.10,
    )
    # The second footprint cell reaches its gas-density receiver limit first.
    np.testing.assert_allclose(limited, 3.0e-3, rtol=0.0, atol=1.0e-15)

    riser_after_old_face = riser_area - dt / dz * (
        old_flux[1:] - old_flux[:-1]
    )
    riser_after = riser_after_old_face.copy()
    riser_after[0] += dt / dz * (-limited - old_flux[0])
    assert riser_after[0] >= 0.0
    np.testing.assert_array_equal(riser_after[1:], riser_after_old_face[1:])

    horizontal_after, discharge_after = _apply_finite_width_side_t_exchange(
        horizontal_area,
        horizontal_discharge,
        upward_flow=-limited,
        opening_weights=weights,
        dt=dt,
        cell_width=dx,
        full_area=horizontal_full,
    )
    riser_loss = float(np.sum(riser_area - riser_after) * dz)
    horizontal_gain = float(
        np.sum(horizontal_after - horizontal_area) * dx
    )
    np.testing.assert_allclose(horizontal_gain, riser_loss, rtol=0.0, atol=1.0e-16)
    np.testing.assert_array_equal(horizontal_after[weights == 0.0], horizontal_area[weights == 0.0])
    np.testing.assert_array_equal(discharge_after, horizontal_discharge)

    minimum_void = horizontal_gas_mass / (2.0 * rho * dx)
    active = weights > 0.0
    assert np.all(
        horizontal_full - horizontal_after[active]
        >= minimum_void[active] - 1.0e-15
    )


def test_smagorinsky_stress_vanishes_for_uniform_translation() -> None:
    full = 0.006
    area = np.full(24, 0.72 * full)
    discharge = area * 0.31
    advanced = _implicit_smagorinsky_momentum_diffusion(
        area,
        discharge,
        full_area=full,
        diameter=0.094,
        spacing=0.01,
        dt=0.02,
        coefficient=0.10,
        molecular_viscosity=0.0,
    )
    np.testing.assert_allclose(advanced, discharge, rtol=0.0, atol=1.0e-15)


def test_smagorinsky_stress_conserves_momentum_and_dissipates_energy() -> None:
    full = 0.006
    area = np.full(31, 0.68 * full)
    x = np.linspace(0.0, 1.0, area.size)
    discharge = area * (0.45 * np.sin(2.0 * np.pi * x))
    energy_before = float(np.sum(0.5 * discharge**2 / area))
    momentum_before = float(np.sum(discharge))
    advanced = _implicit_smagorinsky_momentum_diffusion(
        area,
        discharge,
        full_area=full,
        diameter=0.094,
        spacing=0.01,
        dt=0.05,
        coefficient=0.10,
        molecular_viscosity=0.0,
    )
    energy_after = float(np.sum(0.5 * advanced**2 / area))
    np.testing.assert_allclose(np.sum(advanced), momentum_before, atol=1.0e-15)
    assert energy_after < energy_before


def test_side_t_east_cutcell_conserves_both_phase_inventories() -> None:
    full = math.pi * 0.094**2 / 4.0
    dx = 0.04
    area = np.full(20, full)
    area[:10] = 0.55 * full
    discharge = area * 0.12
    rho = 1.25
    gas_mass = rho * np.maximum(full - area, 1.0e-4 * full) * dx
    gas_momentum = gas_mass * 0.18
    liquid_before = float(np.sum(area) * dx)
    gas_before = float(np.sum(gas_mass))
    momentum_before = float(np.sum(gas_momentum))
    result = _open_side_t_east_capillary_cutcell(
        area,
        discharge,
        gas_mass,
        gas_momentum,
        junction_face=10,
        cell_width=dx,
        full_area=full,
        opening_width=0.0571,
        target_void_fraction=0.00825,
    )
    area_after, _, mass_after, momentum_after, opened, transferred = result
    assert opened > 0.0
    assert transferred > 0.0
    assert area_after[10] < full
    np.testing.assert_allclose(np.sum(area_after) * dx, liquid_before, atol=1.0e-16)
    np.testing.assert_allclose(np.sum(mass_after), gas_before, atol=1.0e-16)
    np.testing.assert_allclose(np.sum(momentum_after), momentum_before, atol=1.0e-16)


def test_side_t_east_cutcell_opens_partially_when_donor_mass_is_limited() -> None:
    full = math.pi * 0.094**2 / 4.0
    dx = 0.02
    area = np.full(24, full)
    area[:12] = 0.92 * full
    discharge = np.zeros_like(area)
    gas_mass = np.zeros_like(area)
    gas_mass[:12] = 1.2 * (full - area[:12]) * dx
    gas_mass[12:] = 1.2 * 1.0e-4 * full * dx
    gas_momentum = 0.2 * gas_mass
    liquid_before = float(np.sum(area) * dx)
    gas_before = float(np.sum(gas_mass))

    area_after, _, mass_after, _, opened, transferred = (
        _open_side_t_east_capillary_cutcell(
            area,
            discharge,
            gas_mass,
            gas_momentum,
            junction_face=12,
            cell_width=dx,
            full_area=full,
            opening_width=0.0571,
            target_void_fraction=0.50,
        )
    )
    geometric_target = 0.5 * full * 0.5 * 0.0571
    assert 0.0 < opened < geometric_target
    assert transferred <= 0.95 * gas_mass[11] * (1.0 + 1.0e-12)
    np.testing.assert_allclose(np.sum(area_after) * dx, liquid_before, atol=1.0e-16)
    np.testing.assert_allclose(np.sum(mass_after), gas_before, atol=1.0e-16)


def test_uniform_stratified_state_has_constant_internal_flux() -> None:
    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    n = 40
    dx = 0.01
    area = np.full(n, 0.76 * area_full)
    discharge = np.zeros(n)
    void = area_full - area
    rho_g = 101325.0 / (287.05 * 293.0)
    gas_mass = np.full(n, rho_g * void * dx)
    gas_momentum = np.zeros(n)
    f1, f2, _, _ = _decoupled_liquid_rusanov_flux(
        area,
        discharge,
        gas_mass,
        gas_momentum,
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=dx,
    )
    np.testing.assert_allclose(f1, 0.0, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(f2, f2[0], rtol=1.0e-12, atol=1.0e-15)


def test_horizontal_ssprk2_preserves_uniform_stratified_rest_state() -> None:
    from casea_coupled_gas_network import CoupledGasParameters

    params = CoupledGasParameters(
        horizontal_diameter=0.094,
        vertical_diameter=0.0571,
    )
    n = 40
    dx = 0.02
    area = np.full(n, 0.72 * params.horizontal_area)
    discharge = np.zeros(n)
    void = params.horizontal_area - area
    gas_mass = np.full(n, params.rho_atmospheric * void[0] * dx)
    gas_momentum = np.zeros(n)

    area_after, q_after, _, _ = (
        _advance_horizontal_liquid_hyperbolic_ssprk2(
            area,
            discharge,
            gas_mass,
            gas_momentum,
            area_full=params.horizontal_area,
            diameter=params.horizontal_diameter,
            wave_speed=45.0,
            cell_width=dx,
            dt=1.0e-4,
            valve_face=10,
            valve_transmissivity=1.0,
            junction_wave_active=True,
            rho_reference=params.rho_atmospheric,
            coupled_gas_parameters=params,
            phase_volume_cfl=1.0,
        )
    )

    np.testing.assert_allclose(area_after, area, atol=1.0e-15)
    np.testing.assert_allclose(q_after, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.sum(area_after), np.sum(area), atol=1.0e-15)


def test_uniform_gas_layer_uses_circular_shallow_water_flux() -> None:
    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    n = 24
    dx = 0.01
    area = np.full(n, 0.72 * area_full)
    discharge = np.zeros(n)
    void = area_full - area
    pressure = 101325.0 + 998.0 * 9.81 * 0.20
    density = pressure / (287.05 * 293.0)
    gas_mass = np.full(n, density * void * dx)
    gas_momentum = np.zeros(n)

    _, momentum_flux, _, _ = _decoupled_liquid_rusanov_flux(
        area,
        discharge,
        gas_mass,
        gas_momentum,
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=dx,
    )

    # A component with no adjacent elastic-liquid front uses the natural
    # circular-section Saint--Venant gauge g*I1.  Gas pressure is a separate
    # network source and therefore does not alter this uniform flux.
    from casea_horizontal_liquid_operator import (
        HorizontalLiquidParameters,
        pressure_potential_state,
    )

    local_params = HorizontalLiquidParameters(
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=dx,
        gravity=9.81,
        rho_liquid=998.0,
        gas_constant=287.05,
        gas_temperature=293.0,
        atmospheric_pressure=101325.0,
        tension_head=0.05,
    )
    expected = pressure_potential_state(
        area[0],
        discharge[0],
        gas_mass[0],
        gas_momentum[0],
        True,
        local_params,
    ).potential
    np.testing.assert_allclose(
        momentum_flux, float(expected), rtol=1.0e-12, atol=1.0e-15
    )


def test_connected_pocket_uses_one_gauge_at_elastic_material_front() -> None:
    """A static gas/full-water contact must not launch a numerical impulse."""

    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    n = 24
    dx = 0.01
    area = np.concatenate(
        (np.full(n // 2, 0.72 * area_full), np.full(n // 2, area_full))
    )
    discharge = np.zeros(n)
    density = (
        101325.0 + 998.0 * 9.81 * 0.20
    ) / (287.05 * 293.0)
    gas_mass = np.concatenate(
        (
            np.full(
                n // 2,
                density * (area_full - 0.72 * area_full) * dx,
            ),
            np.zeros(n // 2),
        )
    )
    volume_flux, momentum_flux, _, _ = _decoupled_liquid_rusanov_flux(
        area,
        discharge,
        gas_mass,
        np.zeros(n),
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=dx,
    )
    interface_face = n // 2
    self_flux = 0.5 * 9.81 * area_full * diameter
    self_tol = 2.0e-14
    assert abs(float(volume_flux[interface_face])) <= self_tol
    np.testing.assert_allclose(
        momentum_flux[interface_face - 1:interface_face + 2],
        self_flux,
        rtol=0.0,
        atol=5.0e-14,
    )


def test_gas_free_riser_projection_preserves_volume_and_momentum() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([full, 0.2 * full, 0.9 * full, 0.0, 0.4 * full])
    discharge = np.array([1.0, -0.2, 0.4, 0.0, 0.1]) * 1.0e-4
    packed, packed_q = _project_single_liquid_column(
        area, discharge, full, dz
    )
    np.testing.assert_allclose(np.sum(packed) * dz, np.sum(area) * dz)
    np.testing.assert_allclose(
        np.sum(packed_q) * dz, np.sum(discharge) * dz
    )
    wet = np.flatnonzero(packed > 0.0)
    assert np.array_equal(wet, np.arange(wet[-1] + 1))
    assert np.count_nonzero((packed > 0.0) & (packed < full)) <= 1


def test_gas_free_riser_projection_keeps_elastic_overfill_volume() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([1.04 * full, full, 0.3 * full, 0.0, 0.0])
    discharge = np.array([1.04, 1.0, 0.3, 0.0, 0.0]) * 1.0e-4
    packed, packed_q = _project_single_liquid_column(
        area, discharge, full, dz
    )
    np.testing.assert_allclose(np.sum(packed) * dz, np.sum(area) * dz)
    np.testing.assert_allclose(
        np.sum(packed_q) * dz, np.sum(discharge) * dz
    )
    assert np.all(packed <= full)


def test_fitted_taylor_core_displaces_liquid_conservatively() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([full, full, full, 0.4 * full, 0.0, 0.0])
    discharge = area * 0.18
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)
    shifted_area, shifted_q, retained, returned = (
        _fit_riser_taylor_core(
            area,
            discharge,
            front_height=0.025,
            gas_core_area_fraction=0.80,
            full_area=full,
            dz=dz,
        )
    )
    assert retained > 0.0
    assert returned == 0.0
    assert shifted_area[0] < area[0]
    assert shifted_area[3] > area[3]
    np.testing.assert_allclose(np.sum(shifted_area) * dz, volume_before)
    np.testing.assert_allclose(np.sum(shifted_q) * dz, momentum_before)


def test_fitted_taylor_core_returns_unstored_liquid_to_junction() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.full(4, full)
    discharge = np.zeros(4)
    shifted_area, _, retained, returned = _fit_riser_taylor_core(
        area,
        discharge,
        front_height=0.04,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )
    assert retained == 0.0
    assert returned > 0.0
    np.testing.assert_allclose(
        np.sum(shifted_area) * dz + returned,
        np.sum(area) * dz,
    )


def test_fitted_taylor_core_is_idempotent_material_topology() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([full, full, full, 0.4 * full, 0.0, 0.0])
    discharge = area * 0.18
    first_area, first_q, retained, returned = _fit_riser_taylor_core(
        area,
        discharge,
        front_height=0.025,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )
    second_area, second_q, retained_again, returned_again = (
        _fit_riser_taylor_core(
            first_area,
            first_q,
            front_height=0.025,
            gas_core_area_fraction=0.80,
            full_area=full,
            dz=dz,
        )
    )
    assert retained > 0.0
    assert returned == 0.0
    assert retained_again == 0.0
    assert returned_again == 0.0
    np.testing.assert_array_equal(second_area, first_area)
    np.testing.assert_array_equal(second_q, first_q)


def test_new_taylor_slice_is_conservative_and_stationary_front_is_idempotent() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([full, full, full, 0.5 * full, 0.0, 0.0])
    discharge = area * 0.12
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)
    opened_area, opened_q, opened_volume = _displace_newly_swept_taylor_slice(
        area,
        discharge,
        old_front_height=0.01,
        new_front_height=0.025,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )
    assert opened_volume > 0.0
    np.testing.assert_allclose(np.sum(opened_area) * dz, volume_before)
    np.testing.assert_allclose(np.sum(opened_q) * dz, momentum_before)
    same_area, same_q, second_opening = _displace_newly_swept_taylor_slice(
        opened_area,
        opened_q,
        old_front_height=0.025,
        new_front_height=0.025,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )
    assert second_opening == 0.0
    np.testing.assert_array_equal(same_area, opened_area)
    np.testing.assert_array_equal(same_q, opened_q)


def test_side_fed_taylor_sweep_returns_liquid_conservatively_to_tee() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([full, full, full, 0.5 * full, 0.0, 0.0])
    discharge = area * 0.12
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)

    swept_area, swept_q, returned, returned_velocity = (
        _return_new_taylor_sweep_to_side_t(
            area,
            discharge,
            old_front_height=0.01,
            new_front_height=0.025,
            gas_core_area_fraction=0.80,
            full_area=full,
            dz=dz,
        )
    )
    expected_return = 0.80 * full * 0.015
    assert math.isclose(returned, expected_return, rel_tol=0.0, abs_tol=1.0e-16)
    assert math.isclose(
        float(np.sum(swept_area) * dz + returned),
        volume_before,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    )
    assert math.isclose(
        float(np.sum(swept_q) * dz + returned * returned_velocity),
        momentum_before,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    )

    horizontal_area = np.full(5, 0.25 * full)
    horizontal_q = np.zeros(5)
    weights = np.array([0.0, 0.25, 0.50, 0.25, 0.0])
    horizontal_volume_before = float(np.sum(horizontal_area) * dz)
    deposited_area, deposited_q = _apply_finite_width_side_t_exchange(
        horizontal_area,
        horizontal_q,
        upward_flow=-returned / 1.0e-3,
        opening_weights=weights,
        dt=1.0e-3,
        cell_width=dz,
    )
    assert math.isclose(
        float(np.sum(deposited_area) * dz),
        horizontal_volume_before + returned,
        rel_tol=0.0,
        abs_tol=1.0e-16,
    )
    np.testing.assert_array_equal(deposited_q, horizontal_q)


def test_twostream_handoff_releases_elastic_storage_locally_and_conservatively() -> None:
    full = 2.0
    dz = 0.1
    area = np.array([2.2, 1.6, 0.0, 1.4, 0.0])
    discharge = np.array([0.44, -0.16, 0.0, 0.07, 0.0])

    mapped_area, mapped_discharge, released = (
        _relax_elastic_riser_storage_for_twostream_handoff(
            area,
            discharge,
            full_area=full,
            dz=dz,
        )
    )

    assert math.isclose(released, 0.02, rel_tol=0.0, abs_tol=1.0e-15)
    np.testing.assert_allclose(mapped_area, [2.0, 1.8, 0.0, 1.4, 0.0])
    np.testing.assert_allclose(mapped_discharge, [0.4, -0.12, 0.0, 0.07, 0.0])
    assert math.isclose(float(np.sum(mapped_area)), float(np.sum(area)))
    assert math.isclose(
        float(np.sum(mapped_discharge)), float(np.sum(discharge))
    )
    # The dry separation at cell 2 prevents the lower compressed parcel from
    # being teleported into the detached upper liquid island.
    assert mapped_area[3] == area[3]


def test_refilled_taylor_core_returns_only_excess_and_respects_receiver_limit() -> None:
    full = 2.0
    dz = 0.1
    area = np.array([1.0, 0.6, 1.8, 0.0])
    discharge = np.array([-0.2, -0.06, 0.18, 0.0])
    # The first cell is fully swept (capacity 0.4); the second is half swept
    # (capacity 1.2); cells above the front are untouched.
    returned_area, returned_q, returned, velocity = (
        _return_refilled_taylor_core_to_side_t(
            area,
            discharge,
            front_height=0.15,
            gas_core_area_fraction=0.8,
            full_area=full,
            dz=dz,
            maximum_return_volume=0.03,
        )
    )

    assert math.isclose(returned, 0.03, rel_tol=0.0, abs_tol=1.0e-15)
    # Requested excess is 0.6*dz, so the receiver cap removes exactly one half.
    np.testing.assert_allclose(returned_area, [0.7, 0.6, 1.8, 0.0])
    np.testing.assert_allclose(returned_q, [-0.14, -0.06, 0.18, 0.0])
    assert math.isclose(velocity, -0.2, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(
        float(np.sum(returned_area) * dz + returned),
        float(np.sum(area) * dz),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )


def test_confined_taylor_refill_is_restored_to_upper_slug_idempotently() -> None:
    full = 2.0
    dz = 0.1
    area = np.array([1.0, 0.6, 1.0, 0.0])
    discharge = np.array([-0.2, -0.06, 0.1, 0.0])
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)

    restored_area, restored_q, moved = (
        _restore_refilled_taylor_core_to_unswept_slug(
            area,
            discharge,
            front_height=0.15,
            gas_core_area_fraction=0.8,
            full_area=full,
            dz=dz,
        )
    )

    assert math.isclose(moved, 0.06, rel_tol=0.0, abs_tol=1.0e-15)
    np.testing.assert_allclose(restored_area, [0.4, 0.6, 1.6, 0.0])
    assert math.isclose(float(np.sum(restored_area) * dz), volume_before)
    assert math.isclose(float(np.sum(restored_q) * dz), momentum_before)
    repeated_area, repeated_q, repeated_moved = (
        _restore_refilled_taylor_core_to_unswept_slug(
            restored_area,
            restored_q,
            front_height=0.15,
            gas_core_area_fraction=0.8,
            full_area=full,
            dz=dz,
        )
    )
    assert repeated_moved == 0.0
    np.testing.assert_array_equal(repeated_area, restored_area)
    np.testing.assert_array_equal(repeated_q, restored_q)


def test_taylor_topology_projection_reconnects_slug_without_losing_momentum() -> None:
    full = 2.5e-3
    dz = 0.01
    area = np.array([0.10, 0.40, 0.20, 0.30, 0.50, 1.00]) * full
    discharge = area * 0.25
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)

    projected_area, projected_q, returned = _project_riser_taylor_topology(
        area,
        discharge,
        front_height=0.025,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )

    assert returned == 0.0
    np.testing.assert_allclose(
        projected_area / full,
        np.array([0.20, 0.20, 0.60, 1.00, 0.50, 0.00]),
    )
    np.testing.assert_allclose(np.sum(projected_area) * dz, volume_before)
    np.testing.assert_allclose(np.sum(projected_q) * dz, momentum_before)
    np.testing.assert_allclose(projected_q, 0.25 * projected_area)

    repeated_area, repeated_q, repeated_return = _project_riser_taylor_topology(
        projected_area,
        projected_q,
        front_height=0.025,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
    )
    assert repeated_return == 0.0
    np.testing.assert_array_equal(repeated_area, projected_area)
    np.testing.assert_array_equal(repeated_q, projected_q)


def test_taylor_topology_uses_drift_film_and_slow_slug_kinematics() -> None:
    full = 2.5e-3
    dz = 0.02
    area = np.array([0.2, 0.4, 1.0, 1.0, 0.0]) * full
    discharge = area * 6.0
    projected_area, projected_q, returned = _project_riser_taylor_topology(
        area,
        discharge,
        front_height=0.04,
        gas_core_area_fraction=0.80,
        full_area=full,
        dz=dz,
        film_velocity=-1.0,
        slug_velocity=0.04,
    )
    assert returned == 0.0
    np.testing.assert_allclose(projected_area[:2], 0.20 * full)
    np.testing.assert_allclose(projected_q[:2] / projected_area[:2], -1.0)
    wet_slug = projected_area[2:] > 0.0
    np.testing.assert_allclose(
        projected_q[2:][wet_slug] / projected_area[2:][wet_slug],
        0.04,
    )


def test_vertical_holdup_head_matches_full_column_and_partial_pool() -> None:
    np.testing.assert_allclose(
        _vertical_liquid_holdup_head(np.ones(3), 0.02),
        np.array([0.05, 0.03, 0.01]),
    )
    np.testing.assert_allclose(
        _vertical_liquid_holdup_head(np.array([0.85, 0.50, 0.07]), 0.02),
        np.array([0.0199, 0.0064, 0.0007]),
    )


def test_annular_film_return_is_volume_and_momentum_conservative() -> None:
    full = 2.5e-3
    dz = 0.01
    rho = 1.2
    area = np.array([0.03, 0.10, 1.0, 1.0, 0.5, 0.0]) * full
    discharge = area * np.array([0.8, 0.7, 0.1, 0.1, -0.1, 0.0])
    tracer = np.array([0.8, 0.7, 0.2, 0.0, 0.0, 0.0]) * rho * full * dz
    volume_before = float(np.sum(area) * dz)
    momentum_before = float(np.sum(discharge) * dz)
    film_area, film_q, returned = _restore_riser_annular_film(
        area,
        discharge,
        tracer,
        minimum_film_fraction=0.20,
        full_area=full,
        dz=dz,
        rho_reference=rho,
    )
    assert returned > 0.0
    assert np.all(film_area[:3] >= 0.20 * full - 1.0e-15)
    np.testing.assert_allclose(np.sum(film_area) * dz, volume_before)
    np.testing.assert_allclose(np.sum(film_q) * dz, momentum_before)


def test_taylor_front_records_swept_height_after_breakthrough() -> None:
    front, velocity, vented = _advance_riser_taylor_front(
        0.34,
        free_surface_height=0.356,
        liquid_superficial_velocity=0.10,
        diameter=0.0571,
        riser_height=0.610,
        dt=0.10,
    )
    assert vented
    assert math.isclose(front, 0.356)
    assert velocity > 0.0

    followed, followed_velocity, still_vented = _advance_riser_taylor_front(
        front,
        free_surface_height=0.31,
        liquid_superficial_velocity=5.0,
        diameter=0.0571,
        riser_height=0.610,
        dt=0.10,
        already_vented=vented,
    )
    assert still_vented
    assert math.isclose(followed, front)
    assert math.isclose(followed_velocity, 0.0)


def test_falling_or_vanishing_surface_cannot_unsweep_taylor_material() -> None:
    for surface in (0.20, 0.0):
        front, velocity, vented = _advance_riser_taylor_front(
            0.28,
            free_surface_height=surface,
            liquid_superficial_velocity=0.0,
            diameter=0.0571,
            riser_height=0.610,
            dt=0.01,
        )
        assert vented
        assert math.isclose(front, 0.28)
        assert math.isclose(velocity, 0.0)


def test_annular_film_is_not_created_above_taylor_front() -> None:
    full = 2.5e-3
    dz = 0.01
    rho = 1.2
    area = np.array([0.05, 0.05, 0.0, 0.8, 0.8]) * full
    discharge = np.zeros_like(area)
    tracer = np.full_like(area, 0.8 * rho * full * dz)
    film_area, _, returned = _restore_riser_annular_film(
        area,
        discharge,
        tracer,
        minimum_film_fraction=0.20,
        full_area=full,
        dz=dz,
        rho_reference=rho,
        front_height=0.02,
    )
    assert returned > 0.0
    assert np.all(film_area[:2] >= 0.20 * full - 1.0e-15)
    assert np.all(film_area[2:] <= area[2:] + 1.0e-15)
    assert film_area[2] == 0.0
    np.testing.assert_allclose(np.sum(film_area), np.sum(area))


def test_vw_film_closure_balances_initial_displacement_and_drainage() -> None:
    diameter = 0.0571
    delta, core_fraction, film_flow, film_velocity = (
        _vw_laminar_film_closure(diameter)
    )
    full_area = 0.25 * math.pi * diameter**2
    core_area = core_fraction * full_area
    terminal_speed = 0.345 * math.sqrt(9.81 * diameter)
    gravity_film_flow = (
        math.pi * 9.81 * diameter * (998.0 - 101325.0 / (287.05 * 293.0))
        * delta**3 / (3.0 * 1.003e-3)
    )
    assert 0.0 < delta < 0.5 * diameter
    assert 0.0 < core_fraction < 1.0
    np.testing.assert_allclose(film_flow, core_area * terminal_speed, rtol=1.0e-12)
    np.testing.assert_allclose(film_flow, gravity_film_flow, rtol=1.0e-12)
    np.testing.assert_allclose(
        film_velocity,
        -film_flow / ((1.0 - core_fraction) * full_area),
        rtol=1.0e-12,
    )


def test_near_dry_momentum_regularization_leaves_resolved_film_unchanged() -> None:
    full = 2.5e-3
    area = np.array([1.0, 0.07, 1.0e-3, 1.0e-5, 0.0]) * full
    discharge = np.full_like(area, 2.0e-4)
    regularized = _regularize_near_dry_momentum(
        area,
        discharge,
        full_area=full,
    )
    assert regularized[0] > 0.999999 * discharge[0]
    assert regularized[1] > 0.999 * discharge[1]
    assert math.isclose(regularized[2], 0.5 * discharge[2])
    assert regularized[3] < 1.0e-3 * discharge[3]
    assert regularized[4] == 0.0


def test_annular_film_friction_rejects_out_of_regime_laminar_terminal_velocity() -> None:
    diameter = 0.0571
    thickness, core_fraction, _, terminal_velocity = _vw_laminar_film_closure(
        diameter
    )
    full_area = 0.25 * math.pi * diameter**2
    film_area = (1.0 - core_fraction) * full_area
    rate_at_laminar_value = _riser_liquid_friction_rate(
        np.array([film_area]),
        np.array([film_area * terminal_velocity]),
        np.array([True]),
        full_area=full_area,
        diameter=diameter,
        film_thickness=thickness,
    )[0]
    reynolds = abs(terminal_velocity) * (2.0 * thickness) / 1.0e-6
    assert reynolds > 4000.0
    # At this Reynolds number a turbulent annular-film stress must exceed
    # gravity by a wide margin; the laminar Nusselt terminal value cannot be a
    # self-consistent boundary flux for Case A.
    assert rate_at_laminar_value * abs(terminal_velocity) > 5.0 * 9.81

    def acceleration_balance(speed: float) -> float:
        rate = _riser_liquid_friction_rate(
            np.array([film_area]),
            np.array([-film_area * speed]),
            np.array([True]),
            full_area=full_area,
            diameter=diameter,
            film_thickness=thickness,
        )[0]
        return rate * speed - 9.81

    lower, upper = 0.0, abs(terminal_velocity)
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if acceleration_balance(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    turbulent_terminal_speed = 0.5 * (lower + upper)
    assert 0.5 < turbulent_terminal_speed < 1.5
    assert turbulent_terminal_speed < 0.5 * abs(terminal_velocity)


def test_side_t_outflow_removes_horizontal_momentum_with_volume() -> None:
    area = 4.0e-3
    discharge = area * 2.5
    area_after, discharge_after = _orthogonal_junction_liquid_exchange(
        area,
        discharge,
        upward_flow=1.0e-3,
        dt=0.02,
        cell_width=0.04,
    )
    assert area_after < area
    assert math.isclose(
        discharge_after / area_after,
        discharge / area,
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )


def test_side_t_return_enters_without_horizontal_momentum() -> None:
    area = 4.0e-3
    discharge = area * 2.5
    area_after, discharge_after = _orthogonal_junction_liquid_exchange(
        area,
        discharge,
        upward_flow=-1.0e-3,
        dt=0.02,
        cell_width=0.04,
    )
    assert area_after > area
    assert discharge_after == discharge


def test_three_branch_return_splits_across_faces_without_node_storage() -> None:
    area = np.full(6, 4.0e-3)
    discharge = np.full(6, 2.0e-4)
    dt = 0.01
    dx = 0.04
    reference = 2.0e-4
    west_flow = -3.0e-4
    east_flow = 7.0e-4
    area_after, discharge_after = (
        _replace_horizontal_face_with_tjunction_fluxes(
            area,
            discharge,
            junction_face=3,
            reference_flow=reference,
            west_flow=west_flow,
            east_flow=east_flow,
            dt=dt,
            cell_width=dx,
        )
    )
    changed = np.flatnonzero(np.abs(area_after - area) > 1.0e-15)
    assert np.array_equal(changed, np.array([2, 3]))
    np.testing.assert_allclose(
        np.sum(area_after - area) * dx,
        (east_flow - west_flow) * dt,
    )
    # Both cells receive orthogonal inflow in this return-flow example; no
    # axial momentum is prescribed by the vertical branch.
    np.testing.assert_array_equal(discharge_after, discharge)


def test_three_branch_flux_has_no_prescribed_wave_pattern() -> None:
    area = np.linspace(2.0e-3, 4.0e-3, 8)
    discharge = np.linspace(-1.0e-4, 3.0e-4, 8)
    area_after, _ = _replace_horizontal_face_with_tjunction_fluxes(
        area,
        discharge,
        junction_face=4,
        reference_flow=1.5e-4,
        west_flow=-2.0e-4,
        east_flow=5.0e-4,
        dt=0.005,
        cell_width=0.02,
    )
    np.testing.assert_array_equal(area_after[:3], area[:3])
    np.testing.assert_array_equal(area_after[5:], area[5:])


def test_three_branch_outflow_carries_only_donor_axial_momentum() -> None:
    area = np.full(6, 4.0e-3)
    discharge = area * np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    area_after, discharge_after = _replace_horizontal_face_with_tjunction_fluxes(
        area,
        discharge,
        junction_face=3,
        reference_flow=0.0,
        west_flow=4.0e-4,
        east_flow=0.0,
        dt=0.01,
        cell_width=0.04,
    )
    assert area_after[2] < area[2]
    np.testing.assert_allclose(
        discharge_after[2] / area_after[2],
        discharge[2] / area[2],
    )
    np.testing.assert_array_equal(discharge_after[3:], discharge[3:])


def test_three_branch_positivity_factor_preserves_vertical_balance() -> None:
    area = np.array([1.0, 0.01, 0.02, 1.0])
    reference = 0.0
    west, east, factor = _limit_three_branch_junction_flows(
        area,
        junction_face=2,
        reference_flow=reference,
        west_flow=2.0,
        east_flow=-2.0,
        dt=0.01,
        cell_width=0.10,
        retained_fraction=0.10,
    )
    assert 0.0 < factor < 1.0
    np.testing.assert_allclose(west - east, factor * 4.0)
    area_after, _ = _replace_horizontal_face_with_tjunction_fluxes(
        area,
        np.zeros_like(area),
        junction_face=2,
        reference_flow=reference,
        west_flow=west,
        east_flow=east,
        dt=0.01,
        cell_width=0.10,
    )
    assert area_after[1] >= 0.10 * area[1] - 1.0e-15
    assert area_after[2] >= 0.10 * area[2] - 1.0e-15


def test_massless_horizontal_void_uses_finite_tension_elastic_branch() -> None:
    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    area = np.full(12, 0.92 * area_full)
    discharge = np.zeros_like(area)
    gas_mass = np.zeros_like(area)
    gas_momentum = np.zeros_like(area)
    _, f2, _, _ = _decoupled_liquid_rusanov_flux(
        area,
        discharge,
        gas_mass,
        gas_momentum,
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=0.01,
    )
    # The consistent operator uses the exact circular full-section moment
    # A_full*D/2; the legacy lookup table differs by O(1e-8) in flux.
    crown_flux = 0.5 * 9.81 * area_full * diameter
    separation_area = area_full * (1.0 - 0.05 * 9.81 / 28.0**2)
    expected = (
        crown_flux
        + 0.5 * 28.0**2
        * (separation_area**2 - area_full**2) / area_full
    )
    np.testing.assert_allclose(f2, expected, rtol=1.0e-12, atol=1.0e-15)


def test_elastic_pressure_flux_is_continuous_at_pipe_crown() -> None:
    diameter = 0.094
    area_full = math.pi * diameter**2 / 4.0
    area = np.full(12, (1.0 - 1.0e-9) * area_full)
    discharge = np.zeros_like(area)
    _, f2, _, _ = _decoupled_liquid_rusanov_flux(
        area,
        discharge,
        np.zeros_like(area),
        np.zeros_like(area),
        area_full=area_full,
        diameter=diameter,
        wave_speed=28.0,
        cell_width=0.01,
    )
    crown_flux = 0.5 * 9.81 * area_full * diameter
    np.testing.assert_allclose(f2, crown_flux, rtol=0.0, atol=1.0e-8)


def test_gas_void_closure_limiter_is_local_and_conservative() -> None:
    full = 1.0
    dx = 0.1
    dt = 0.01
    area = np.array([0.6, 0.9, 0.6])
    rho = 1.2
    gas_mass = np.array([0.0, rho * 0.1 * dx, 0.0])
    volume_flux = np.array([0.0, 2.0, 0.0, 0.0])
    momentum_flux = 3.0 * volume_flux
    limited_q, limited_f2 = _limit_gas_void_closure_flux(
        area,
        gas_mass,
        volume_flux,
        momentum_flux,
        full_area=full,
        cell_width=dx,
        dt=dt,
        rho_reference=rho,
        density_fraction=0.5,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        closure_fraction=0.25,
    )
    area_new = area - dt / dx * (limited_q[1:] - limited_q[:-1])
    assert area_new[1] <= 0.925 + 1.0e-14
    np.testing.assert_allclose(limited_f2, 3.0 * limited_q)
    assert math.isclose(
        float(np.sum(area_new)), float(np.sum(area)), abs_tol=1.0e-14
    )
    assert limited_q[2] == volume_flux[2]


def test_compressed_gas_still_protects_void_from_liquid_closure() -> None:
    full = 1.0
    dx = 0.1
    dt = 0.01
    area = np.array([0.6, 0.9, 0.6])
    rho = 1.2
    # Five-atmosphere equivalent mass deliberately exceeds the gas momentum
    # solver's resolved-density ceiling.  Positivity must still preserve void.
    gas_mass = np.array([0.0, 5.0 * rho * 0.1 * dx, 0.0])
    volume_flux = np.array([0.0, 2.0, 0.0, 0.0])
    limited_q, _ = _limit_gas_void_closure_flux(
        area,
        gas_mass,
        volume_flux,
        3.0 * volume_flux,
        full_area=full,
        cell_width=dx,
        dt=dt,
        rho_reference=rho,
        density_fraction=0.5,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        closure_fraction=0.25,
    )
    area_new = area - dt / dx * (limited_q[1:] - limited_q[:-1])
    assert area_new[1] <= 0.925 + 1.0e-14


def test_gas_void_limiter_enforces_mass_supported_density_ceiling() -> None:
    full = 1.0
    dx = 0.1
    dt = 0.01
    rho = 1.2
    area = np.array([0.6, 0.9, 0.6])
    gas_mass = np.array([0.0, rho * 0.1 * dx, 0.0])
    volume_flux = np.array([0.0, 2.0, 0.0, 0.0])
    limited_q, _ = _limit_gas_void_closure_flux(
        area,
        gas_mass,
        volume_flux,
        volume_flux,
        full_area=full,
        cell_width=dx,
        dt=dt,
        rho_reference=rho,
        density_fraction=0.5,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        closure_fraction=0.90,
    )
    area_new = area - dt / dx * (limited_q[1:] - limited_q[:-1])
    assert area_new[1] <= 0.95 + 1.0e-14


def test_gas_void_limiter_keeps_open_material_cell_in_gas_topology() -> None:
    full = 1.0
    dx = 0.1
    dt = 0.01
    rho = 1.2
    area = np.array([0.5, 0.999, 0.5])
    gas_mass = np.array([0.0, rho * 0.001 * dx, 0.0])
    volume_flux = np.array([0.0, 1.0, 0.0, 0.0])
    limited_q, _ = _limit_gas_void_closure_flux(
        area,
        gas_mass,
        volume_flux,
        volume_flux,
        full_area=full,
        cell_width=dx,
        dt=dt,
        rho_reference=rho,
        density_fraction=0.02,
        density_ceiling=2.0,
        void_floor_fraction=1.0e-4,
        active_void_fraction=5.0e-4,
        closure_fraction=1.0,
    )
    area_new = area - dt / dx * (limited_q[1:] - limited_q[:-1])
    assert full - area_new[1] >= 5.0e-4 - 1.0e-14


def test_liquid_donor_limiter_accounts_for_both_outflow_faces() -> None:
    area = np.array([0.2, 0.1, 0.3])
    flux = np.array([-0.5, 0.7, 0.8, 0.9])
    momentum = 2.0 * flux
    limited_q, limited_f2 = _limit_liquid_donor_flux(
        area,
        flux,
        momentum,
        cell_width=0.1,
        dt=0.02,
        retained_fraction=0.10,
    )
    area_new = area - 0.02 / 0.1 * (
        limited_q[1:] - limited_q[:-1]
    )
    assert np.all(area_new >= 0.10 * area - 1.0e-14)
    np.testing.assert_allclose(limited_f2, 2.0 * limited_q)
