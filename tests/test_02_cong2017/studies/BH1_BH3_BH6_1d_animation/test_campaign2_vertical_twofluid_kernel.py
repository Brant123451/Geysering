from __future__ import annotations

from dataclasses import replace
import inspect
import math

import pytest

from campaign2_vertical_twofluid_kernel import (
    AtmosphericTopBoundary,
    COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY,
    FirstBottomGasIntrusionTransactionLike,
    MINIMUM_FIRST_BOTTOM_GAS_INTRUSION_FIELDS,
    MISSING_PHYSICAL_CLOSURES,
    StateAdmissibilityError,
    TeeTransactionRejected,
    UPPER_INTERFACE_GEOMETRY_ROUNDOFF_ULPS,
    VERTICAL_TWOFLUID_KERNEL_READY,
    VerticalTwoFluidParameters,
    VerticalTwoFluidState,
    advance_vertical_twofluid,
    apply_equal_and_opposite_interphase_drag,
    canonicalize_upper_free_surface_roundoff,
    hydrostatic_column_state,
    isothermal_common_pressure_faces,
    isothermal_gas_pressure_cells,
    lower_material_front_geometric_timestep_limit,
    lower_material_front_star_state,
)
from case1_persistent_coupling import TeeTransaction
from campaign2_tee_riemann import GasTrace, solve_gas_tee


def _parameters(
    *,
    cells: int = 4,
    dz: float = 0.25,
    diameter: float = 0.041,
    gravity: float = 9.81,
) -> VerticalTwoFluidParameters:
    return VerticalTwoFluidParameters(
        cell_count=cells,
        cell_length_m=dz,
        diameter_m=diameter,
        gravity_m_s2=gravity,
    )


def _mixed_rest_state(
    parameters: VerticalTwoFluidParameters,
    *,
    liquid_fraction: float = 0.25,
) -> VerticalTwoFluidState:
    area = parameters.full_area_m2
    al = liquid_fraction * area
    ag = area - al
    mg = (
        parameters.atmospheric_gas_density_kg_m3
        * ag
        * parameters.cell_length_m
    )
    return VerticalTwoFluidState.from_iterables(
        Al=[al] * parameters.cell_count,
        Ql=[0.0] * parameters.cell_count,
        Mg=[mg] * parameters.cell_count,
        Jg=[0.0] * parameters.cell_count,
    )


def _uniform_lower_front_state(
    parameters: VerticalTwoFluidParameters,
    *,
    front_cell: int,
    front_liquid_fraction: float,
    velocity_m_s: float,
    pressure_abs_Pa: float | None = None,
    upper_first_gas_cell: int | None = None,
) -> VerticalTwoFluidState:
    """Build one exact gas/lower-front/liquid-plug/top-gas component."""

    area = parameters.full_area_m2
    dz = parameters.cell_length_m
    pressure = (
        parameters.atmospheric_pressure_Pa
        if pressure_abs_Pa is None
        else float(pressure_abs_Pa)
    )
    rho_g = pressure / (
        parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    )
    top_start = (
        parameters.cell_count - 2
        if upper_first_gas_cell is None
        else int(upper_first_gas_cell)
    )
    Al = []
    Ql = []
    Mg = []
    Jg = []
    for cell in range(parameters.cell_count):
        if cell < front_cell:
            al = 0.0
        elif cell == front_cell:
            al = front_liquid_fraction * area
        elif cell < top_start:
            al = area
        else:
            al = 0.0
        ag = area - al
        mg = rho_g * ag * dz
        Al.append(al)
        Ql.append(al * velocity_m_s if al > 0.0 else 0.0)
        Mg.append(mg)
        Jg.append(mg * velocity_m_s if mg > 0.0 else 0.0)
    return VerticalTwoFluidState.from_iterables(
        Al=Al,
        Ql=Ql,
        Mg=Mg,
        Jg=Jg,
        lower_material_front_cell=front_cell,
        lower_material_front_orientation="gas_below_liquid_above",
    )


def _finite_pocket_transaction(
    parameters: VerticalTwoFluidParameters,
    *,
    velocity_m_s: float,
    donor_density_kg_m3: float | None = None,
    mass_flow_kg_s: float | None = None,
) -> TeeTransaction:
    area = parameters.full_area_m2
    density = (
        parameters.atmospheric_gas_density_kg_m3
        if donor_density_kg_m3 is None
        else float(donor_density_kg_m3)
    )
    volume_flow = area * velocity_m_s
    mass_flow = (
        density * volume_flow
        if mass_flow_kg_s is None
        else float(mass_flow_kg_s)
    )
    donor_volume_flow = mass_flow / density
    donor_velocity = velocity_m_s
    gas_open_area = abs(donor_volume_flow / donor_velocity)
    assert gas_open_area <= area
    return TeeTransaction(
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=mass_flow,
        gas_volume_flow_to_riser_m3_s=donor_volume_flow,
        gas_normal_momentum_flow_N=mass_flow * donor_velocity,
        liquid_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=12_345.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
        riser_mouth_area_m2=area,
        gas_open_area_m2=gas_open_area,
        liquid_open_area_m2=0.0,
        blocked_riser_area_m2=area - gas_open_area,
    )


def _complete_tee_transaction(
    parameters: VerticalTwoFluidParameters,
    **fields: float | None,
) -> TeeTransaction:
    """Fill the authoritative mouth partition for generic kernel tests."""

    area = parameters.full_area_m2
    mdot = float(fields.get("gas_mass_flow_to_riser_kg_s", 0.0) or 0.0)
    qg = fields.get("gas_volume_flow_to_riser_m3_s")
    pi_g = float(fields.get("gas_normal_momentum_flow_N", 0.0) or 0.0)
    gas_open = 0.0
    if mdot != 0.0 and qg is not None and pi_g != 0.0:
        gas_open = abs(float(qg) / (pi_g / mdot))
    west = float(fields.get("west_liquid_flow_m3_s", 0.0) or 0.0)
    east = float(fields.get("east_liquid_flow_m3_s", 0.0) or 0.0)
    ql = west - east
    pi_l_raw = fields.get("liquid_normal_momentum_flow_N")
    if ql != 0.0 and pi_l_raw is not None and float(pi_l_raw) > 0.0:
        liquid_open = (
            parameters.liquid_density_kg_m3
            * ql
            * ql
            / float(pi_l_raw)
        )
    else:
        liquid_open = area - gas_open
    fields.update(
        riser_mouth_area_m2=area,
        gas_open_area_m2=gas_open,
        liquid_open_area_m2=liquid_open,
        blocked_riser_area_m2=area - gas_open - liquid_open,
    )
    return TeeTransaction(**fields)


def test_grid_aligned_hydrostatic_column_is_a_zero_flow_state() -> None:
    parameters = _parameters(cells=4, dz=0.25)
    # The interface lies exactly between cells 1 and 2: cells are pure liquid
    # below and pure atmospheric gas above, so each phase is well balanced.
    state = hydrostatic_column_state(parameters, liquid_height_m=0.50)

    result = advance_vertical_twofluid(state, parameters, dt=0.01)

    assert result.state.Al == pytest.approx(state.Al, abs=2.0e-18)
    assert result.state.Mg == pytest.approx(state.Mg, abs=2.0e-18)
    assert result.state.Ql == pytest.approx([0.0] * 4, abs=2.0e-17)
    assert result.state.Jg == pytest.approx([0.0] * 4, abs=2.0e-17)
    assert result.top_liquid_outflow_m3_s == 0.0
    assert result.top_gas_mass_flux_kg_s == 0.0
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_published_061_m_grid_aligned_column_is_strictly_quiescent() -> None:
    parameters = _parameters(cells=180, dz=0.01, diameter=0.026)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.61)

    assert all(
        al == pytest.approx(parameters.full_area_m2, abs=2.0e-18)
        for al in state.Al[:61]
    )
    assert all(al == pytest.approx(0.0, abs=2.0e-18) for al in state.Al[61:])

    result = advance_vertical_twofluid(state, parameters, dt=1.0e-4)

    assert result.state.Al == pytest.approx(state.Al, abs=2.0e-18)
    assert result.state.Mg == pytest.approx(state.Mg, abs=2.0e-18)
    assert result.state.Ql == pytest.approx([0.0] * 180, abs=2.0e-17)
    assert result.state.Jg == pytest.approx([0.0] * 180, abs=2.0e-17)
    assert result.top_gas_mass_flux_kg_s == 0.0
    assert result.top_liquid_outflow_m3_s == 0.0


def test_isothermal_eos_pressure_is_closed_from_gas_mass_and_void() -> None:
    parameters = _parameters(cells=2, dz=0.20, gravity=0.0)
    area = parameters.full_area_m2
    al = 0.30 * area
    ag = area - al
    densities = (0.75, 2.25)
    state = VerticalTwoFluidState.from_iterables(
        Al=[al, al],
        Ql=[0.0, 0.0],
        Mg=[rho * ag * parameters.cell_length_m for rho in densities],
        Jg=[0.0, 0.0],
    )

    pressure = isothermal_gas_pressure_cells(state, parameters)
    rt = parameters.gas_constant_J_kg_K * parameters.gas_temperature_K
    assert pressure == pytest.approx([rho * rt for rho in densities])
    faces = isothermal_common_pressure_faces(state, parameters)
    assert faces[1] == pytest.approx(0.5 * (pressure[0] + pressure[1]))
    assert faces[0] == pytest.approx(pressure[0])
    assert faces[-1] == pytest.approx(parameters.atmospheric_pressure_Pa)


def test_twice_atmospheric_pure_gas_column_vents_through_characteristic() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    area = parameters.full_area_m2
    rho = 2.0 * parameters.atmospheric_gas_density_kg_m3
    state = VerticalTwoFluidState.from_iterables(
        Al=[0.0],
        Ql=[0.0],
        Mg=[rho * area * parameters.cell_length_m],
        Jg=[0.0],
    )
    dt = 1.0e-6

    result = advance_vertical_twofluid(state, parameters, dt=dt)

    assert result.top_gas_mass_flux_kg_s > 0.0
    assert result.state.Mg[0] < state.Mg[0]
    assert state.Mg[0] - result.state.Mg[0] == pytest.approx(
        result.top_gas_mass_flux_kg_s * dt, abs=2.0e-18
    )
    assert result.state.cumulative_top_gas_outflow_kg == pytest.approx(
        result.top_gas_mass_flux_kg_s * dt, abs=2.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_bottom_liquid_node_pressure_produces_physical_upward_impulse() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    area = parameters.full_area_m2
    state = VerticalTwoFluidState.from_iterables(
        Al=[area],
        Ql=[0.0],
        Mg=[0.0],
        Jg=[0.0],
    )
    gauge = 2_500.0
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=gauge,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    result = advance_vertical_twofluid(
        state, parameters, dt=1.0e-5, tee_transaction=transaction
    )

    assert result.pressure_faces_Pa[0] == pytest.approx(
        parameters.atmospheric_pressure_Pa + gauge
    )
    assert result.pressure_faces_Pa[-1] == parameters.atmospheric_pressure_Pa
    assert result.state.Ql[0] > 0.0


def test_bottom_gas_interface_pressure_is_applied_only_as_face_pressure() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    area = parameters.full_area_m2
    state = VerticalTwoFluidState.from_iterables(
        Al=[0.0],
        Ql=[0.0],
        Mg=[
            parameters.atmospheric_gas_density_kg_m3
            * area
            * parameters.cell_length_m
        ],
        Jg=[0.0],
    )
    imposed = 1.05 * parameters.atmospheric_pressure_Pa
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=imposed,
    )

    result = advance_vertical_twofluid(
        state, parameters, dt=1.0e-6, tee_transaction=transaction
    )

    assert result.pressure_faces_Pa[0] == pytest.approx(imposed)
    assert result.gas_momentum_flux_faces_N[0] == 0.0
    assert result.state.Jg[0] > 0.0


def test_gas_transaction_cannot_enter_a_zero_void_cell() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    state = VerticalTwoFluidState.from_iterables(
        Al=[parameters.full_area_m2],
        Ql=[0.0],
        Mg=[0.0],
        Jg=[0.0],
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=1.0e-6,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    with pytest.raises(TeeTransactionRejected, match="gas exchange"):
        advance_vertical_twofluid(
            state, parameters, dt=1.0e-4, tee_transaction=transaction
        )


def test_same_mass_momentum_pressure_but_different_donor_density_moves_front_differently() -> None:
    """The vertical displacement follows stored donor Qg, never p/(RT)."""

    parameters = _parameters(cells=4, dz=0.10, diameter=0.041, gravity=0.0)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    mass_flux = 2.0e-6
    momentum_flux = 6.0e-6
    velocity = momentum_flux / mass_flux
    donor_densities = (0.80, 1.60)
    volume_fluxes = tuple(mass_flux / rho for rho in donor_densities)
    open_areas = tuple(volume_flux / velocity for volume_flux in volume_fluxes)
    dt = 1.0e-4

    results = []
    for density, volume_flux, open_area in zip(
        donor_densities, volume_fluxes, open_areas
    ):
        assert density * volume_flux == pytest.approx(mass_flux)
        assert open_area * velocity == pytest.approx(volume_flux)
        assert mass_flux * velocity == pytest.approx(momentum_flux)
        transaction = TeeTransaction(
            west_liquid_flow_m3_s=0.0,
            east_liquid_flow_m3_s=0.0,
            gas_mass_flow_to_riser_kg_s=mass_flux,
            gas_volume_flow_to_riser_m3_s=volume_flux,
            gas_normal_momentum_flow_N=momentum_flux,
            liquid_normal_momentum_flow_N=0.0,
            liquid_node_gauge_pressure_Pa=0.0,
            gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
            riser_mouth_area_m2=parameters.full_area_m2,
            gas_open_area_m2=open_area,
            liquid_open_area_m2=parameters.full_area_m2 - open_area,
            blocked_riser_area_m2=0.0,
        )
        result = advance_vertical_twofluid(
            state,
            parameters,
            dt=dt,
            tee_transaction=transaction,
        )
        results.append(result)
        ledger = result.first_bottom_gas_intrusion
        assert ledger is not None
        assert ledger.donor_gas_density_kg_m3 == pytest.approx(density)
        assert ledger.gas_open_area_m2 == pytest.approx(open_area)
        assert ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=2.0e-22)
        assert ledger.gas_momentum_residual_kg_m_s == pytest.approx(
            0.0, abs=2.0e-22
        )
        assert result.state.Al[0] == pytest.approx(
            parameters.full_area_m2 - dt / parameters.cell_length_m * volume_flux,
            abs=2.0e-18,
        )

    assert volume_fluxes[0] != volume_fluxes[1]
    assert open_areas[0] != open_areas[1]
    assert results[0].state.Al[0] != results[1].state.Al[0]
    assert tuple(
        FirstBottomGasIntrusionTransactionLike.__annotations__
    ) == MINIMUM_FIRST_BOTTOM_GAS_INTRUSION_FIELDS


def test_positive_first_intrusion_missing_independent_fluxes_is_rejected_atomically() -> None:
    """Legacy mdot/Pi/p alone remains insufficient and cannot seed a void."""

    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=1.0e-6,
        gas_normal_momentum_flow_N=1.0e-6,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    snapshot = (state.Al, state.Ql, state.Mg, state.Jg, state.time_s)

    with pytest.raises(
        TeeTransactionRejected,
        match="gas_volume_flow_to_riser_m3_s",
    ):
        advance_vertical_twofluid(
            state,
            parameters,
            dt=1.0e-4,
            tee_transaction=transaction,
        )
    assert (state.Al, state.Ql, state.Mg, state.Jg, state.time_s) == snapshot


def test_first_intrusion_ale_identities_pressure_pair_and_restart_marker() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041, gravity=0.0)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    qg = 2.0e-6
    donor_density = 1.20
    donor_velocity = 3.0
    dt = 1.0e-4
    transaction = TeeTransaction(
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=donor_density * qg,
        gas_volume_flow_to_riser_m3_s=qg,
        gas_normal_momentum_flow_N=donor_density * qg * donor_velocity,
        liquid_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
        riser_mouth_area_m2=parameters.full_area_m2,
        gas_open_area_m2=qg / donor_velocity,
        liquid_open_area_m2=(
            parameters.full_area_m2 - qg / donor_velocity
        ),
        blocked_riser_area_m2=0.0,
    )

    result = advance_vertical_twofluid(
        state, parameters, dt=dt, tee_transaction=transaction
    )
    ledger = result.first_bottom_gas_intrusion
    assert ledger is not None
    assert parameters.full_area_m2 - result.state.Al[0] == pytest.approx(
        dt * qg / parameters.cell_length_m, abs=2.0e-18
    )
    assert result.liquid_volume_flux_faces_m3_s[1] == pytest.approx(
        transaction.liquid_flow_to_riser_m3_s + qg, abs=2.0e-18
    )
    assert result.state.Mg[0] == pytest.approx(
        dt * transaction.gas_mass_flow_to_riser_kg_s, abs=2.0e-18
    )
    assert ledger.gas_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-22
    )
    assert ledger.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=5.0e-20
    )
    assert ledger.mixture_volume_residual_m3 == 0.0
    assert ledger.paired_pressure_impulse_residual_kg_m_s == 0.0
    expected_interface_impulse = (
        ledger.common_pressure_abs_Pa * parameters.full_area_m2 * dt
    )
    assert ledger.liquid_pressure_impulse_kg_m_s == pytest.approx(
        expected_interface_impulse
    )
    assert ledger.gas_pressure_impulse_kg_m_s == pytest.approx(
        -expected_interface_impulse
    )
    assert ledger.liquid_open_area_m2 > 0.0
    assert ledger.gas_open_area_m2 + ledger.liquid_open_area_m2 == pytest.approx(
        ledger.riser_mouth_area_m2
    )
    assert ledger.blocked_riser_area_m2 == 0.0
    assert result.budget.liquid_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-20
    )
    assert result.budget.gas_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-20
    )
    assert result.budget.total_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-20
    )
    assert result.state.lower_material_front_cell == 0
    assert result.state.lower_material_front_orientation == "gas_below_liquid_above"

    restarted = VerticalTwoFluidState.from_iterables(
        Al=result.state.Al,
        Ql=result.state.Ql,
        Mg=result.state.Mg,
        Jg=result.state.Jg,
        time_s=result.state.time_s,
        cumulative_top_liquid_outflow_m3=result.state.cumulative_top_liquid_outflow_m3,
        cumulative_top_gas_outflow_kg=result.state.cumulative_top_gas_outflow_kg,
        cumulative_top_gas_inflow_kg=result.state.cumulative_top_gas_inflow_kg,
        cumulative_bottom_liquid_exchange_m3=result.state.cumulative_bottom_liquid_exchange_m3,
        cumulative_bottom_gas_exchange_kg=result.state.cumulative_bottom_gas_exchange_kg,
        lower_material_front_cell=result.state.lower_material_front_cell,
        lower_material_front_orientation=result.state.lower_material_front_orientation,
    )
    assert restarted == result.state
    with pytest.raises(
        TeeTransactionRejected,
        match="exactly blocked liquid riser opening",
    ):
        advance_vertical_twofluid(
            restarted,
            parameters,
            dt=dt,
            # Reusing the first-entry partition would reconnect liquid through
            # a bottom face now owned by the finite gas pocket.
            tee_transaction=transaction,
        )
    finite_transaction = replace(
        transaction,
        liquid_open_area_m2=0.0,
        blocked_riser_area_m2=(
            parameters.full_area_m2 - transaction.gas_open_area_m2
        ),
    )
    continued = advance_vertical_twofluid(
        restarted,
        parameters,
        dt=dt,
        tee_transaction=finite_transaction,
    )
    assert continued.lower_material_front is not None
    assert continued.bottom_gas_storage is not None
    assert continued.state.lower_material_front_cell == 0
    assert restarted == result.state


def test_unrepresentable_nextafter_and_overlarge_first_entry_roll_back() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041, gravity=0.0)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    snapshot = state
    tiny = math.nextafter(0.0, math.inf)
    tiny_transaction = TeeTransaction(
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=tiny,
        gas_volume_flow_to_riser_m3_s=tiny,
        gas_normal_momentum_flow_N=tiny,
        liquid_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
        riser_mouth_area_m2=parameters.full_area_m2,
        gas_open_area_m2=tiny,
        liquid_open_area_m2=parameters.full_area_m2 - tiny,
        blocked_riser_area_m2=0.0,
    )
    with pytest.raises(TeeTransactionRejected, match="not representable"):
        advance_vertical_twofluid(
            state,
            parameters,
            dt=1.0e-4,
            tee_transaction=tiny_transaction,
        )
    assert state == snapshot

    qg = parameters.full_area_m2
    overlarge = replace(
        tiny_transaction,
        gas_mass_flow_to_riser_kg_s=qg,
        gas_volume_flow_to_riser_m3_s=qg,
        gas_normal_momentum_flow_N=qg,
        gas_open_area_m2=parameters.full_area_m2,
        liquid_open_area_m2=0.0,
    )
    with pytest.raises(TeeTransactionRejected, match="interface CFL"):
        advance_vertical_twofluid(
            state,
            parameters,
            dt=math.nextafter(parameters.cell_length_m, math.inf),
            tee_transaction=overlarge,
        )
    assert state == snapshot


def test_zero_gas_transaction_keeps_the_no_intrusion_path_bitwise() -> None:
    parameters = _parameters(
        cells=4,
        dz=0.10,
        diameter=0.041,
        gravity=0.0,
    )
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    zero = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    without_transaction = advance_vertical_twofluid(
        state, parameters, dt=1.0e-4
    )
    with_zero_transaction = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-4,
        tee_transaction=zero,
    )

    assert with_zero_transaction == without_transaction


def test_mixed_mouth_rejects_incompatible_phase_pressures() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    state = _mixed_rest_state(parameters, liquid_fraction=0.50)
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=100.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    with pytest.raises(TeeTransactionRejected, match="incompatible"):
        advance_vertical_twofluid(
            state, parameters, dt=1.0e-5, tee_transaction=transaction
        )


def test_one_tee_transaction_is_received_once_and_in_equal_amount() -> None:
    parameters = _parameters(cells=2, dz=0.20, gravity=0.0)
    state = _mixed_rest_state(parameters, liquid_fraction=0.15)
    q_to_riser = 0.8e-6
    gas_to_riser = 1.7e-6
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=1.0e-6,
        east_liquid_flow_m3_s=0.2e-6,
        gas_mass_flow_to_riser_kg_s=gas_to_riser,
        gas_normal_momentum_flow_N=0.0,
    )
    dt = 0.004
    liquid_before = math.fsum(state.Al) * parameters.cell_length_m
    gas_before = math.fsum(state.Mg)

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
    )

    liquid_after = math.fsum(result.state.Al) * parameters.cell_length_m
    gas_after = math.fsum(result.state.Mg)
    assert result.bottom_liquid_exchange_m3_s == pytest.approx(q_to_riser)
    assert result.bottom_gas_exchange_kg_s == pytest.approx(gas_to_riser)
    assert liquid_after - liquid_before == pytest.approx(
        q_to_riser * dt, abs=2.0e-18
    )
    assert gas_after - gas_before == pytest.approx(
        gas_to_riser * dt, abs=2.0e-18
    )
    assert result.state.cumulative_bottom_liquid_exchange_m3 == pytest.approx(
        q_to_riser * dt, abs=2.0e-18
    )
    assert result.state.cumulative_bottom_gas_exchange_kg == pytest.approx(
        gas_to_riser * dt, abs=2.0e-18
    )


def test_atmospheric_top_never_backflows_liquid() -> None:
    parameters = _parameters(cells=1, dz=0.20, gravity=0.0)
    base = _mixed_rest_state(parameters, liquid_fraction=0.30)
    state = VerticalTwoFluidState.from_iterables(
        Al=base.Al,
        Ql=[-2.0e-6],
        Mg=base.Mg,
        Jg=base.Jg,
    )
    initial_volume = state.Al[0] * parameters.cell_length_m

    result = advance_vertical_twofluid(state, parameters, dt=0.002)

    assert result.top_liquid_outflow_m3_s == 0.0
    assert result.liquid_volume_flux_faces_m3_s[-1] == 0.0
    assert result.state.cumulative_top_liquid_outflow_m3 == 0.0
    assert result.state.Al[0] * parameters.cell_length_m == pytest.approx(
        initial_volume, abs=2.0e-18
    )


def test_two_phase_mass_budgets_and_real_top_integrals_close() -> None:
    parameters = _parameters(cells=2, dz=0.20, gravity=0.0)
    area = parameters.full_area_m2
    al = 0.25 * area
    ag = area - al
    ql = 2.0e-6
    ug = 0.30
    mg = (
        parameters.atmospheric_gas_density_kg_m3
        * ag
        * parameters.cell_length_m
    )
    gas_flux = ug * mg / parameters.cell_length_m
    state = VerticalTwoFluidState.from_iterables(
        Al=[al, al],
        Ql=[ql, ql],
        Mg=[mg, mg],
        Jg=[mg * ug, mg * ug],
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=ql,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=gas_flux,
        gas_normal_momentum_flow_N=gas_flux * ug,
    )
    dt = 0.003

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
    )

    assert result.top_liquid_outflow_m3_s == pytest.approx(ql)
    assert result.top_gas_mass_flux_kg_s == pytest.approx(gas_flux)
    assert result.state.cumulative_top_liquid_outflow_m3 == pytest.approx(
        ql * dt, abs=2.0e-18
    )
    assert result.state.cumulative_top_gas_outflow_kg == pytest.approx(
        gas_flux * dt, abs=2.0e-18
    )
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=3.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=3.0e-18
    )
    assert (
        result.budget.final_liquid_volume_m3
        - result.budget.initial_liquid_volume_m3
    ) == pytest.approx(
        result.budget.bottom_liquid_exchange_m3
        - result.budget.top_liquid_outflow_m3,
        abs=3.0e-18,
    )
    assert (
        result.budget.final_gas_mass_kg
        - result.budget.initial_gas_mass_kg
    ) == pytest.approx(
        result.budget.bottom_gas_exchange_kg
        - result.budget.top_gas_outflow_kg
        + result.budget.top_gas_inflow_kg,
        abs=3.0e-18,
    )


def test_drag_reduces_slip_and_is_exactly_equal_and_opposite() -> None:
    parameters = _parameters(cells=1, dz=0.10, gravity=0.0)
    area = parameters.full_area_m2
    al = 0.45 * area
    ag = area - al
    ul = -0.20
    ug = 2.0
    rho_g = 2.0
    mg = rho_g * ag * parameters.cell_length_m
    state = VerticalTwoFluidState.from_iterables(
        Al=[al],
        Ql=[al * ul],
        Mg=[mg],
        Jg=[mg * ug],
    )
    initial_total = (
        parameters.liquid_density_kg_m3
        * state.Ql[0]
        * parameters.cell_length_m
        + state.Jg[0]
    )

    updated, ledger = apply_equal_and_opposite_interphase_drag(
        state, parameters, dt=0.05
    )
    final_ul = updated.Ql[0] / updated.Al[0]
    final_ug = updated.Jg[0] / updated.Mg[0]
    final_total = (
        parameters.liquid_density_kg_m3
        * updated.Ql[0]
        * parameters.cell_length_m
        + updated.Jg[0]
    )

    assert abs(final_ug - final_ul) < abs(ug - ul)
    assert ledger.total_gas_impulse_kg_m_s != 0.0
    assert ledger.total_liquid_impulse_kg_m_s == pytest.approx(
        -ledger.total_gas_impulse_kg_m_s, abs=2.0e-18
    )
    assert ledger.exchange_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert final_total == pytest.approx(initial_total, abs=2.0e-18)


def test_finite_pocket_intracell_translation_closes_storage_and_global_budgets() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = 0.02
    state = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.60,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    transaction = _finite_pocket_transaction(
        parameters, velocity_m_s=1.5 * velocity
    )
    dt = 0.01
    result = advance_vertical_twofluid(
        state, parameters, dt=dt, tee_transaction=transaction
    )
    q = parameters.full_area_m2 * velocity
    assert result.state.Al[0] == pytest.approx(
        state.Al[0] - q * dt / parameters.cell_length_m,
        abs=3.0e-18,
    )
    assert result.state.lower_material_front_cell == 0
    assert result.bottom_gas_storage is not None
    assert result.lower_material_front is not None
    imposed_gas_flow = 1.5 * q
    assert (
        result.bottom_gas_storage.gas_volume_flow_m3_s
        == pytest.approx(imposed_gas_flow)
    )
    assert result.bottom_gas_storage.gas_pocket_volume_change_m3 == pytest.approx(
        q * dt, abs=4.0e-18
    )
    storage = result.bottom_gas_storage.compressive_storage_volume_m3
    expected_storage = (
        result.bottom_gas_storage.gas_volume_flow_m3_s
        - result.bottom_gas_storage.lower_front_volume_flow_m3_s
    ) * dt
    assert storage == expected_storage
    assert result.bottom_gas_storage.gas_pocket_geometry_residual_m3 == pytest.approx(
        0.0,
        abs=4.0e-18,
    )
    assert abs(storage) > 1.0e-12
    assert result.lower_material_front.liquid_plug_volume_residual_m3 == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert result.lower_material_front.gas_component_mass_residual_kg == pytest.approx(
        0.0, abs=5.0e-18
    )
    assert result.lower_material_front.paired_pressure_impulse_residual_kg_m_s == 0.0
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=5.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=5.0e-18
    )
    assert result.budget.total_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert result.budget.liquid_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert result.budget.gas_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-15
    )


def test_finite_pocket_bottom_pressure_respects_open_and_blocked_areas() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    state = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.60,
        velocity_m_s=0.0,
        upper_first_gas_cell=6,
    )
    area = parameters.full_area_m2
    imposed = 120_000.0

    def zero_flow_transaction(
        gas_open_area: float,
        interface_pressure: float = imposed,
    ) -> TeeTransaction:
        return TeeTransaction(
            west_liquid_flow_m3_s=0.0,
            east_liquid_flow_m3_s=0.0,
            gas_mass_flow_to_riser_kg_s=0.0,
            gas_volume_flow_to_riser_m3_s=0.0,
            gas_normal_momentum_flow_N=0.0,
            liquid_normal_momentum_flow_N=0.0,
            liquid_node_gauge_pressure_Pa=0.0,
            gas_interface_pressure_abs_Pa=interface_pressure,
            riser_mouth_area_m2=area,
            gas_open_area_m2=gas_open_area,
            liquid_open_area_m2=0.0,
            blocked_riser_area_m2=area - gas_open_area,
        )

    closed = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-5,
        tee_transaction=zero_flow_transaction(0.0),
    )
    closed_reference = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-5,
        tee_transaction=zero_flow_transaction(
            0.0,
            parameters.atmospheric_pressure_Pa,
        ),
    )
    assert closed.pressure_faces_Pa[0] == pytest.approx(
        parameters.atmospheric_pressure_Pa
    )
    assert closed.pressure_faces_Pa == closed_reference.pressure_faces_Pa
    assert closed.state == closed_reference.state

    gas_open_area = 0.25 * area
    partial = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-5,
        tee_transaction=zero_flow_transaction(gas_open_area),
    )
    expected = (
        gas_open_area * imposed
        + (area - gas_open_area) * parameters.atmospheric_pressure_Pa
    ) / area
    assert partial.pressure_faces_Pa[0] == pytest.approx(expected)
    assert partial.state.Jg[0] > 0.0


def test_lower_front_lands_exactly_on_face_then_crosses_multiple_cells_by_events() -> None:
    parameters = _parameters(cells=9, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = 0.02
    state = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.20,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    transaction = _finite_pocket_transaction(parameters, velocity_m_s=velocity)
    visited = [state.lower_material_front_cell]
    for _ in range(4):
        dt = lower_material_front_geometric_timestep_limit(
            state, parameters, cfl=1.0
        )
        result = advance_vertical_twofluid(
            state, parameters, dt=dt, tee_transaction=transaction
        )
        state = result.state
        visited.append(state.lower_material_front_cell)
        marker = state.lower_material_front_cell
        if marker is not None and state.Al[marker] == parameters.full_area_m2:
            assert result.lower_material_front is not None
            assert result.lower_material_front.new_grid_aligned is True

    assert visited[1] == 1
    assert max(cell for cell in visited if cell is not None) >= 2
    marker = state.lower_material_front_cell
    assert marker is not None
    assert all(state.Al[cell] == 0.0 for cell in range(marker))


def test_lower_front_reversal_can_remove_the_bottom_pocket_without_seed_or_loss() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = -0.02
    state = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.80,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    transaction = _finite_pocket_transaction(parameters, velocity_m_s=velocity)
    dt = lower_material_front_geometric_timestep_limit(
        state, parameters, cfl=1.0
    )
    result = advance_vertical_twofluid(
        state, parameters, dt=dt, tee_transaction=transaction
    )
    assert result.state.lower_material_front_cell is None
    assert result.state.lower_material_front_orientation is None
    assert result.state.Al[0] == parameters.full_area_m2
    assert result.state.Mg[0] == 0.0
    assert result.state.Jg[0] == 0.0
    assert result.lower_material_front is not None
    assert result.lower_material_front.new_front_cell is None
    assert result.lower_material_front.gas_pocket_mass_change_kg == pytest.approx(
        result.lower_material_front.bottom_gas_mass_exchange_kg,
        abs=5.0e-18,
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=5.0e-18
    )


def test_lower_front_restart_is_topology_complete_and_deterministic() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = 0.015
    initial = _uniform_lower_front_state(
        parameters,
        front_cell=1,
        front_liquid_fraction=0.55,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    transaction = _finite_pocket_transaction(parameters, velocity_m_s=velocity)
    first = advance_vertical_twofluid(
        initial, parameters, dt=0.01, tee_transaction=transaction
    )
    restarted = VerticalTwoFluidState.from_iterables(
        Al=first.state.Al,
        Ql=first.state.Ql,
        Mg=first.state.Mg,
        Jg=first.state.Jg,
        time_s=first.state.time_s,
        cumulative_top_liquid_outflow_m3=first.state.cumulative_top_liquid_outflow_m3,
        cumulative_top_gas_outflow_kg=first.state.cumulative_top_gas_outflow_kg,
        cumulative_top_gas_inflow_kg=first.state.cumulative_top_gas_inflow_kg,
        cumulative_bottom_liquid_exchange_m3=first.state.cumulative_bottom_liquid_exchange_m3,
        cumulative_bottom_gas_exchange_kg=first.state.cumulative_bottom_gas_exchange_kg,
        lower_material_front_cell=first.state.lower_material_front_cell,
        lower_material_front_orientation=first.state.lower_material_front_orientation,
    )
    direct = advance_vertical_twofluid(
        first.state, parameters, dt=0.01, tee_transaction=transaction
    )
    from_restart = advance_vertical_twofluid(
        restarted, parameters, dt=0.01, tee_transaction=transaction
    )
    assert from_restart == direct


def test_finite_storage_keeps_donor_volume_independent_from_receiver_eos() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = 0.01
    state = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.50,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    area = parameters.full_area_m2
    rho_low, rho_high = 0.8, 1.6
    common_mass_flow = 0.40 * rho_low * area * velocity
    results = []
    for density in (rho_low, rho_high):
        transaction = _finite_pocket_transaction(
            parameters,
            velocity_m_s=velocity,
            donor_density_kg_m3=density,
            mass_flow_kg_s=common_mass_flow,
        )
        results.append(
            advance_vertical_twofluid(
                state, parameters, dt=0.01, tee_transaction=transaction
            )
        )
    low, high = results
    assert low.state.Al == high.state.Al
    assert low.state.Mg == high.state.Mg
    assert low.state.Ql != high.state.Ql
    assert low.state.Jg != high.state.Jg
    assert low.bottom_gas_storage is not None
    assert high.bottom_gas_storage is not None
    assert low.bottom_gas_storage.donor_gas_density_kg_m3 == pytest.approx(rho_low)
    assert high.bottom_gas_storage.donor_gas_density_kg_m3 == pytest.approx(rho_high)
    assert low.bottom_gas_storage.gas_volume_flow_m3_s == pytest.approx(
        2.0 * high.bottom_gas_storage.gas_volume_flow_m3_s
    )
    assert abs(
        low.bottom_gas_storage.compressive_storage_volume_m3
        - high.bottom_gas_storage.compressive_storage_volume_m3
    ) > 1.0e-20


def test_lower_front_reference_pressure_shift_and_oversized_step_rollback() -> None:
    parameters = _parameters(cells=8, dz=0.10, diameter=0.041, gravity=0.0)
    velocity = 0.02
    base = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.20,
        velocity_m_s=velocity,
        upper_first_gas_cell=6,
    )
    shifted = _uniform_lower_front_state(
        parameters,
        front_cell=0,
        front_liquid_fraction=0.20,
        velocity_m_s=velocity,
        pressure_abs_Pa=parameters.atmospheric_pressure_Pa + 20_000.0,
        upper_first_gas_cell=6,
    )
    base_star = lower_material_front_star_state(base, parameters)
    shifted_star = lower_material_front_star_state(
        shifted,
        parameters,
        AtmosphericTopBoundary(
            pressure_abs_Pa=parameters.atmospheric_pressure_Pa + 20_000.0
        ),
    )
    assert shifted_star.interface_velocity_m_s == pytest.approx(
        base_star.interface_velocity_m_s, abs=2.0e-15
    )
    assert shifted_star.interface_pressure_abs_Pa == pytest.approx(
        base_star.interface_pressure_abs_Pa + 20_000.0,
        abs=2.0e-9,
    )
    transaction = _finite_pocket_transaction(parameters, velocity_m_s=velocity)
    crossing = lower_material_front_geometric_timestep_limit(
        base, parameters, cfl=1.0
    )
    with pytest.raises(TeeTransactionRejected, match="lower material-interface CFL"):
        advance_vertical_twofluid(
            base,
            parameters,
            dt=1.01 * crossing,
            tee_transaction=transaction,
        )
    assert base.lower_material_front_cell == 0
    assert base.Al[0] == pytest.approx(0.20 * parameters.full_area_m2)


def test_scope_is_explicit_and_no_result_feedback_enters_the_step() -> None:
    assert VERTICAL_TWOFLUID_KERNEL_READY is True
    assert COMPLETE_CAMPAIGN2_VERTICAL_CLOSURE_READY is False
    assert (
        "post_first_entry_finite_t_mouth_two_phase_riemann_continuation"
        not in MISSING_PHYSICAL_CLOSURES
    )
    assert (
        "multi_cell_lower_gas_liquid_interface_propagation"
        not in MISSING_PHYSICAL_CLOSURES
    )
    assert set(inspect.signature(advance_vertical_twofluid).parameters) == {
        "state",
        "parameters",
        "dt",
        "tee_transaction",
        "top_boundary",
        "pressure_faces_Pa",
    }


def test_equal_pressure_open_gas_interface_cannot_inject_momentum() -> None:
    parameters = _parameters(cells=2, dz=0.20, gravity=0.0)
    state = _mixed_rest_state(parameters, liquid_fraction=0.50)
    density = parameters.atmospheric_gas_density_kg_m3
    trace = GasTrace(
        parameters.atmospheric_pressure_Pa,
        density,
        0.0,
        345.0,
    )
    gas = solve_gas_tee(
        trace,
        trace,
        open_area_m2=0.50 * parameters.full_area_m2,
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=gas.mass_flow_to_riser_kg_s,
        gas_normal_momentum_flow_N=gas.normal_momentum_flow_N,
        gas_interface_pressure_abs_Pa=gas.interface_pressure_abs_Pa,
    )

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-5,
        tee_transaction=transaction,
    )

    assert result.state.Jg == pytest.approx([0.0, 0.0], abs=2.0e-18)
    assert result.state.Ql == pytest.approx([0.0, 0.0], abs=2.0e-18)
    assert result.pressure_faces_Pa[0] == pytest.approx(
        parameters.atmospheric_pressure_Pa
    )
    assert result.budget.final_total_momentum_kg_m_s == pytest.approx(
        0.0, abs=2.0e-18
    )


def test_unresolved_gas_area_cannot_accumulate_source_momentum() -> None:
    # Reproduce the former t=0.1484 s source-stage failure using its exact
    # diameter, cell length, residual gas area, face-pressure difference and
    # dt.  Mg is identically zero.  Before the fix, each call added about
    # 4.30e-17 kg m/s to Jg until validate_state rejected the fifth call.
    parameters = VerticalTwoFluidParameters(
        cell_count=1,
        cell_length_m=0.01,
        diameter_m=0.026,
        liquid_density_kg_m3=998.2,
    )
    residual_gas_area = 2.1937746758071697e-15
    state = VerticalTwoFluidState.from_iterables(
        Al=[parameters.full_area_m2 - residual_gas_area],
        Ql=[0.0],
        Mg=[0.0],
        Jg=[0.0],
    )
    pressure_drop = -97.92341883041081
    pressure_faces = (
        parameters.atmospheric_pressure_Pa - pressure_drop,
        parameters.atmospheric_pressure_Pa,
    )

    for _ in range(8):
        result = advance_vertical_twofluid(
            state,
            parameters,
            dt=2.0e-4,
            pressure_faces_Pa=pressure_faces,
        )
        state = result.state
        assert state.Mg[0] == 0.0
        assert state.Jg[0] == 0.0


def test_near_pure_cells_keep_only_resolved_phase_source_momentum() -> None:
    parameters = _parameters(cells=1, dz=0.01, diameter=0.026)
    area = parameters.full_area_m2
    pressure_faces = (
        parameters.atmospheric_pressure_Pa + 100.0,
        parameters.atmospheric_pressure_Pa,
    )
    top = AtmosphericTopBoundary(allow_gas_inflow=False)

    # A sub-resolution liquid sliver has no liquid velocity degree of freedom,
    # while the resolved gas inventory must retain its real pressure impulse.
    unresolved_liquid_area = 0.5 * parameters.area_tolerance_m2
    gas_area = area - unresolved_liquid_area
    gas_mass = (
        parameters.atmospheric_gas_density_kg_m3
        * gas_area
        * parameters.cell_length_m
    )
    nearly_pure_gas = VerticalTwoFluidState.from_iterables(
        Al=[unresolved_liquid_area],
        Ql=[0.0],
        Mg=[gas_mass],
        Jg=[0.0],
    )

    gas_result = advance_vertical_twofluid(
        nearly_pure_gas,
        parameters,
        dt=1.0e-6,
        top_boundary=top,
        pressure_faces_Pa=pressure_faces,
    )

    assert gas_result.state.Ql[0] == 0.0
    assert gas_result.state.Jg[0] > 0.0

    # Conversely, a resolved gas pocket only just above both resolution bounds
    # must not be mistaken for the absent-gas branch or lose its pressure force.
    resolved_gas_area = 2.0 * parameters.area_tolerance_m2
    resolved_gas_mass = max(
        parameters.atmospheric_gas_density_kg_m3
        * resolved_gas_area
        * parameters.cell_length_m,
        2.0 * parameters.mass_tolerance_kg,
    )
    nearly_pure_liquid = VerticalTwoFluidState.from_iterables(
        Al=[area - resolved_gas_area],
        Ql=[0.0],
        Mg=[resolved_gas_mass],
        Jg=[0.0],
    )

    liquid_result = advance_vertical_twofluid(
        nearly_pure_liquid,
        parameters,
        dt=1.0e-6,
        top_boundary=top,
        pressure_faces_Pa=pressure_faces,
    )

    assert liquid_result.state.Jg[0] > 0.0


def test_positive_gas_uses_one_eos_and_velocity_across_tolerances() -> None:
    parameters = VerticalTwoFluidParameters(
        cell_count=2,
        cell_length_m=0.01,
        diameter_m=0.026,
        gravity_m_s2=0.0,
        interphase_drag_multiplier=0.0,
    )
    area = parameters.full_area_m2
    requested_gas_areas = (
        0.5 * parameters.area_tolerance_m2,
        2.0 * parameters.area_tolerance_m2,
    )
    liquid_areas = tuple(area - gas_area for gas_area in requested_gas_areas)
    gas_areas = tuple(area - liquid_area for liquid_area in liquid_areas)
    density = parameters.atmospheric_gas_density_kg_m3
    velocity = 0.50
    gas_masses = tuple(
        density * gas_area * parameters.cell_length_m
        for gas_area in gas_areas
    )
    assert gas_masses[0] < parameters.mass_tolerance_kg
    assert gas_masses[1] > parameters.mass_tolerance_kg
    state = VerticalTwoFluidState.from_iterables(
        Al=liquid_areas,
        Ql=[liquid_area * velocity for liquid_area in liquid_areas],
        Mg=gas_masses,
        Jg=[mass * velocity for mass in gas_masses],
    )

    pressure = isothermal_gas_pressure_cells(state, parameters)
    assert pressure == pytest.approx(
        [parameters.atmospheric_pressure_Pa] * 2,
        rel=2.0e-12,
    )
    dragged, _ = apply_equal_and_opposite_interphase_drag(
        state, parameters, dt=1.0e-4
    )
    assert dragged.Jg == pytest.approx(state.Jg, rel=0.0, abs=1.0e-30)
    assert tuple(jg / mg for jg, mg in zip(dragged.Jg, dragged.Mg)) == (
        velocity,
        velocity,
    )


def test_exact_gas_inventory_invariants_keep_only_zero_mass_roundoff_void() -> None:
    parameters = _parameters(cells=1, dz=0.01, diameter=0.026)
    area = parameters.full_area_m2

    with pytest.raises(StateAdmissibilityError, match="no positive volume"):
        isothermal_gas_pressure_cells(
            VerticalTwoFluidState.from_iterables(
                Al=[area], Ql=[0.0], Mg=[1.0e-30], Jg=[0.0]
            ),
            parameters,
        )
    with pytest.raises(StateAdmissibilityError, match="zero-mass gas"):
        isothermal_gas_pressure_cells(
            VerticalTwoFluidState.from_iterables(
                Al=[area - 0.5 * parameters.area_tolerance_m2],
                Ql=[0.0],
                Mg=[0.0],
                Jg=[1.0e-30],
            ),
            parameters,
        )
    with pytest.raises(StateAdmissibilityError, match="resolved gas volume"):
        isothermal_gas_pressure_cells(
            VerticalTwoFluidState.from_iterables(
                Al=[area - 2.0 * parameters.area_tolerance_m2],
                Ql=[0.0],
                Mg=[0.0],
                Jg=[0.0],
            ),
            parameters,
        )

    roundoff_void = VerticalTwoFluidState.from_iterables(
        Al=[area - 0.5 * parameters.area_tolerance_m2],
        Ql=[0.0],
        Mg=[0.0],
        Jg=[0.0],
    )
    assert isothermal_gas_pressure_cells(roundoff_void, parameters) == (None,)


def test_upper_interface_roundoff_canonicalization_is_ulp_only_and_conservative_for_gas() -> None:
    parameters = _parameters(
        cells=4, dz=0.10, diameter=0.041, gravity=0.0
    )
    area = parameters.full_area_m2
    rho = parameters.atmospheric_gas_density_kg_m3
    resolved = hydrostatic_column_state(parameters, liquid_height_m=0.15)
    assert canonicalize_upper_free_surface_roundoff(
        resolved, parameters
    ) is resolved

    tiny_liquid = 0.5 * (
        UPPER_INTERFACE_GEOMETRY_ROUNDOFF_ULPS * math.ulp(area)
    )
    gas_areas = (0.0, area - tiny_liquid, area, area)
    exhausted_liquid = VerticalTwoFluidState.from_iterables(
        Al=[area, tiny_liquid, 0.0, 0.0],
        Ql=[0.0] * 4,
        Mg=[rho * ag * parameters.cell_length_m for ag in gas_areas],
        Jg=[0.0] * 4,
    )
    liquid_pinned = canonicalize_upper_free_surface_roundoff(
        exhausted_liquid, parameters
    )
    assert liquid_pinned.Al[1] == 0.0
    assert liquid_pinned.Mg == exhausted_liquid.Mg

    one_ulp_gas_area = math.ulp(area)
    liquid_area = area - one_ulp_gas_area
    stored_gas_area = area - liquid_area
    exhausted_gas = VerticalTwoFluidState.from_iterables(
        Al=[area, liquid_area, 0.0, 0.0],
        Ql=[0.0] * 4,
        Mg=[
            0.0,
            rho * stored_gas_area * parameters.cell_length_m,
            rho * area * parameters.cell_length_m,
            rho * area * parameters.cell_length_m,
        ],
        Jg=[0.0, 2.0e-22, 0.0, 0.0],
    )
    gas_before = math.fsum(exhausted_gas.Mg)
    momentum_before = math.fsum(exhausted_gas.Jg)
    gas_pinned = canonicalize_upper_free_surface_roundoff(
        exhausted_gas, parameters
    )
    assert gas_pinned.Al[1] == area
    assert gas_pinned.Mg[1] == 0.0
    assert gas_pinned.Jg[1] == 0.0
    assert math.fsum(gas_pinned.Mg) == pytest.approx(
        gas_before, rel=0.0, abs=2.0e-30
    )
    assert math.fsum(gas_pinned.Jg) == pytest.approx(
        momentum_before, rel=0.0, abs=2.0e-30
    )


def test_saturated_projection_keeps_a_static_column_quiescent() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)

    result = advance_vertical_twofluid(state, parameters, dt=1.0e-4)

    assert result.liquid_volume_flux_faces_m3_s[:3] == (0.0, 0.0, 0.0)
    assert result.state.Al == pytest.approx(state.Al, abs=2.0e-18)
    assert result.state.Ql[:2] == pytest.approx((0.0, 0.0), abs=2.0e-17)
    assert result.state.Mg[:2] == (0.0, 0.0)
    assert result.state.Jg[:2] == (0.0, 0.0)


def test_saturated_volume_flux_reaches_surface_and_conserves_liquid() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(state, parameters)
    imposed_flow = 5.0e-7
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    initial_volume = math.fsum(state.Al) * parameters.cell_length_m

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
    )

    assert result.liquid_volume_flux_faces_m3_s[:3] == pytest.approx(
        (imposed_flow, imposed_flow, imposed_flow), abs=2.0e-20
    )
    assert result.state.Al[:2] == pytest.approx(
        (parameters.full_area_m2, parameters.full_area_m2), abs=2.0e-18
    )
    assert all(
        parameters.full_area_m2 - result.state.Al[cell]
        <= parameters.area_tolerance_m2
        for cell in (0, 1)
    )
    assert result.state.Mg[:2] == (0.0, 0.0)
    assert result.state.Jg[:2] == (0.0, 0.0)
    assert result.state.Al[2] > state.Al[2]
    final_volume = math.fsum(result.state.Al) * parameters.cell_length_m
    assert final_volume - initial_volume == pytest.approx(
        imposed_flow * dt, abs=3.0e-18
    )
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=3.0e-18
    )


def test_positive_flux_rewets_grid_aligned_surface_and_displaces_real_donor_gas() -> None:
    parameters = _parameters(
        cells=4, dz=0.10, diameter=0.041, gravity=0.0
    )
    rest = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    donor_velocity = 0.40
    state = VerticalTwoFluidState.from_iterables(
        Al=rest.Al,
        Ql=rest.Ql,
        Mg=rest.Mg,
        Jg=[0.0, 0.0, rest.Mg[2] * donor_velocity, 0.0],
    )
    base_faces = isothermal_common_pressure_faces(state, parameters)
    imposed_flow = 1.0e-7
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    initial_liquid_volume = math.fsum(state.Al) * parameters.cell_length_m
    initial_gas_mass = math.fsum(state.Mg)
    gas_volume = parameters.full_area_m2 * parameters.cell_length_m
    donor_density = state.Mg[2] / gas_volume
    swept_volume = imposed_flow * dt
    displaced_mass = donor_density * swept_volume

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )

    assert result.liquid_volume_flux_faces_m3_s[:4] == pytest.approx(
        (imposed_flow, imposed_flow, imposed_flow, 0.0), abs=2.0e-20
    )
    assert result.gas_mass_flux_faces_kg_s[2] == 0.0
    assert result.gas_mass_flux_faces_kg_s[3] == pytest.approx(
        donor_density * imposed_flow, abs=2.0e-20
    )
    assert result.gas_momentum_flux_faces_N[3] == pytest.approx(
        donor_density * imposed_flow * donor_velocity, abs=2.0e-20
    )
    assert result.state.Al[2] == pytest.approx(
        swept_volume / parameters.cell_length_m, abs=2.0e-18
    )
    assert state.Mg[2] - result.state.Mg[2] == pytest.approx(
        displaced_mass, abs=2.0e-18
    )
    assert result.state.Mg[3] - state.Mg[3] == pytest.approx(
        displaced_mass, abs=2.0e-18
    )
    assert math.fsum(result.state.Mg) == pytest.approx(
        initial_gas_mass, abs=3.0e-18
    )
    assert (
        math.fsum(result.state.Al) * parameters.cell_length_m
        - initial_liquid_volume
    ) == pytest.approx(swept_volume, abs=3.0e-18)

    ledger = result.upper_free_surface_advance
    assert ledger is not None
    assert (ledger.interface_cell, ledger.interface_face) == (2, 3)
    assert ledger.swept_liquid_volume_m3 == pytest.approx(swept_volume)
    assert ledger.donor_gas_density_kg_m3 == pytest.approx(donor_density)
    assert ledger.donor_gas_velocity_m_s == pytest.approx(donor_velocity)
    assert ledger.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert ledger.donor_gas_mass_residual_kg == pytest.approx(
        0.0, abs=3.0e-18
    )
    assert ledger.donor_gas_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=3.0e-18
    )
    assert ledger.liquid_pressure_impulse_kg_m_s == pytest.approx(
        -ledger.gas_pressure_impulse_kg_m_s
    )
    assert ledger.paired_pressure_impulse_residual_kg_m_s == 0.0
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=4.0e-18
    )


def test_retreat_then_advance_reversal_uses_same_cut_cell_after_restart() -> None:
    parameters = _parameters(
        cells=4, dz=0.10, diameter=0.041, gravity=0.0
    )
    initial = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(initial, parameters)
    magnitude = 1.0e-7
    dt = 1.0e-4

    def transaction(flow: float) -> TeeTransaction:
        return _complete_tee_transaction(
            parameters,
            west_liquid_flow_m3_s=flow,
            east_liquid_flow_m3_s=0.0,
            gas_mass_flow_to_riser_kg_s=0.0,
            gas_normal_momentum_flow_N=0.0,
            liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
            gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
        )

    retreated = advance_vertical_twofluid(
        initial,
        parameters,
        dt=dt,
        tee_transaction=transaction(-magnitude),
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )
    restarted = VerticalTwoFluidState.from_iterables(
        Al=retreated.state.Al,
        Ql=retreated.state.Ql,
        Mg=retreated.state.Mg,
        Jg=retreated.state.Jg,
        time_s=retreated.state.time_s,
        cumulative_top_liquid_outflow_m3=(
            retreated.state.cumulative_top_liquid_outflow_m3
        ),
        cumulative_top_gas_outflow_kg=(
            retreated.state.cumulative_top_gas_outflow_kg
        ),
        cumulative_top_gas_inflow_kg=(
            retreated.state.cumulative_top_gas_inflow_kg
        ),
        cumulative_bottom_liquid_exchange_m3=(
            retreated.state.cumulative_bottom_liquid_exchange_m3
        ),
        cumulative_bottom_gas_exchange_kg=(
            retreated.state.cumulative_bottom_gas_exchange_kg
        ),
    )
    advance_magnitude = 0.5 * magnitude
    advanced = advance_vertical_twofluid(
        restarted,
        parameters,
        dt=dt,
        tee_transaction=transaction(advance_magnitude),
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )

    assert retreated.upper_free_surface_retreat is not None
    assert retreated.upper_free_surface_retreat.interface_cell == 1
    assert advanced.upper_free_surface_advance is not None
    assert advanced.upper_free_surface_advance.interface_cell == 1
    assert advanced.gas_mass_flux_faces_kg_s[1] == 0.0
    assert advanced.liquid_volume_flux_faces_m3_s[2] == 0.0
    assert retreated.state.Al[1] < advanced.state.Al[1] < initial.Al[1]
    assert 0.0 < advanced.state.Mg[1] < retreated.state.Mg[1]
    assert math.fsum(advanced.state.Mg) == pytest.approx(
        math.fsum(initial.Mg), abs=4.0e-18
    )


def test_microscopic_advance_survives_restart_without_threshold_dead_zone() -> None:
    parameters = _parameters(
        cells=4, dz=0.10, diameter=0.041, gravity=0.0
    )
    initial = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(initial, parameters)
    imposed_flow = 1.0e-14
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    first = advance_vertical_twofluid(
        initial, parameters, dt=dt, tee_transaction=transaction
    )
    restarted = VerticalTwoFluidState.from_iterables(
        Al=first.state.Al,
        Ql=first.state.Ql,
        Mg=first.state.Mg,
        Jg=first.state.Jg,
        time_s=first.state.time_s,
        cumulative_top_liquid_outflow_m3=(
            first.state.cumulative_top_liquid_outflow_m3
        ),
        cumulative_top_gas_outflow_kg=first.state.cumulative_top_gas_outflow_kg,
        cumulative_top_gas_inflow_kg=first.state.cumulative_top_gas_inflow_kg,
        cumulative_bottom_liquid_exchange_m3=(
            first.state.cumulative_bottom_liquid_exchange_m3
        ),
        cumulative_bottom_gas_exchange_kg=(
            first.state.cumulative_bottom_gas_exchange_kg
        ),
    )
    second = advance_vertical_twofluid(
        restarted, parameters, dt=dt, tee_transaction=transaction
    )

    assert 0.0 < first.state.Al[2] < parameters.area_tolerance_m2
    assert first.upper_free_surface_advance is not None
    assert second.upper_free_surface_advance is not None
    assert first.upper_free_surface_advance.interface_cell == 2
    assert second.upper_free_surface_advance.interface_cell == 2
    assert second.state.Al[2] == pytest.approx(
        2.0 * imposed_flow * dt / parameters.cell_length_m,
        abs=2.0e-18,
    )


def test_bottom_overpressure_accelerates_saturated_column_uniformly() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(state, parameters)
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=0.0,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(
            base_faces[0] - base_faces[2] + 500.0
        ),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=1.0e-4,
        tee_transaction=transaction,
    )

    assert result.state.Al == pytest.approx(state.Al, abs=2.0e-18)
    assert result.state.Ql[0] > 0.0
    assert result.state.Ql[0] == pytest.approx(
        result.state.Ql[1], rel=2.0e-13, abs=2.0e-18
    )
    assert result.pressure_faces_Pa[0] == pytest.approx(
        base_faces[0] + 500.0
    )
    assert result.pressure_faces_Pa[2] == pytest.approx(base_faces[2])


def test_negative_flux_retracts_a_sharp_surface_conservatively() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    rest = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    donor_velocity = -0.40
    state = VerticalTwoFluidState.from_iterables(
        Al=rest.Al,
        Ql=rest.Ql,
        Mg=rest.Mg,
        Jg=[0.0, 0.0, rest.Mg[2] * donor_velocity, rest.Mg[3] * donor_velocity],
    )
    base_faces = isothermal_common_pressure_faces(state, parameters)
    imposed_flow = -1.0e-7
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    initial_liquid_volume = math.fsum(state.Al) * parameters.cell_length_m
    initial_gas_mass = math.fsum(state.Mg)
    donor_density = state.Mg[2] / (
        parameters.full_area_m2 * parameters.cell_length_m
    )
    swept_volume = -imposed_flow * dt
    transferred_mass = donor_density * swept_volume

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )

    assert result.liquid_volume_flux_faces_m3_s[:3] == pytest.approx(
        (imposed_flow, imposed_flow, 0.0), abs=2.0e-20
    )
    assert result.gas_mass_flux_faces_kg_s[2] == pytest.approx(
        donor_density * imposed_flow, abs=2.0e-20
    )
    assert result.gas_momentum_flux_faces_N[2] == pytest.approx(
        donor_density * imposed_flow * donor_velocity, abs=2.0e-20
    )
    assert result.state.Al[0] == pytest.approx(state.Al[0], abs=2.0e-18)
    assert result.state.Al[1] == pytest.approx(
        state.Al[1] - swept_volume / parameters.cell_length_m,
        abs=2.0e-18,
    )
    assert result.state.Mg[1] == pytest.approx(transferred_mass, abs=2.0e-18)
    assert math.fsum(result.state.Mg) == pytest.approx(
        initial_gas_mass, abs=2.0e-18
    )
    assert (
        math.fsum(result.state.Al) * parameters.cell_length_m
        - initial_liquid_volume
    ) == pytest.approx(imposed_flow * dt, abs=3.0e-18)

    ledger = result.upper_free_surface_retreat
    assert ledger is not None
    assert (ledger.interface_cell, ledger.interface_face) == (1, 2)
    assert ledger.swept_gas_volume_m3 == pytest.approx(swept_volume)
    assert ledger.donor_gas_density_kg_m3 == pytest.approx(donor_density)
    assert ledger.donor_gas_velocity_m_s == pytest.approx(donor_velocity)
    assert ledger.liquid_volume_residual_m3 == pytest.approx(0.0, abs=4.0e-18)
    assert ledger.receiver_gas_mass_residual_kg == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert ledger.receiver_gas_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert ledger.liquid_pressure_impulse_kg_m_s == pytest.approx(
        -ledger.gas_pressure_impulse_kg_m_s
    )
    assert ledger.paired_pressure_impulse_residual_kg_m_s == 0.0
    assert result.budget.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert result.budget.gas_mass_residual_kg == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert result.budget.total_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=4.0e-17
    )


def test_microscopic_retreat_has_no_threshold_dead_zone() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(state, parameters)
    imposed_flow = -1.0e-14
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    result = advance_vertical_twofluid(
        state, parameters, dt=dt, tee_transaction=transaction
    )

    created_void_area = parameters.full_area_m2 - result.state.Al[1]
    assert 0.0 < created_void_area < parameters.area_tolerance_m2
    assert 0.0 < result.state.Mg[1] < parameters.mass_tolerance_kg
    assert result.upper_free_surface_retreat is not None
    assert result.upper_free_surface_retreat.interface_cell == 1
    assert math.fsum(result.state.Mg) == pytest.approx(
        math.fsum(state.Mg), abs=2.0e-20
    )


def test_partial_cut_cell_retreat_continues_on_the_same_interface() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    initial = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    base_faces = isothermal_common_pressure_faces(initial, parameters)
    imposed_flow = -1.0e-7
    dt = 1.0e-4
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=imposed_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    first = advance_vertical_twofluid(
        initial,
        parameters,
        dt=dt,
        tee_transaction=transaction,
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )
    restarted = VerticalTwoFluidState.from_iterables(
        Al=first.state.Al,
        Ql=first.state.Ql,
        Mg=first.state.Mg,
        Jg=first.state.Jg,
        time_s=first.state.time_s,
        cumulative_top_liquid_outflow_m3=(
            first.state.cumulative_top_liquid_outflow_m3
        ),
        cumulative_top_gas_outflow_kg=(
            first.state.cumulative_top_gas_outflow_kg
        ),
        cumulative_top_gas_inflow_kg=(
            first.state.cumulative_top_gas_inflow_kg
        ),
        cumulative_bottom_liquid_exchange_m3=(
            first.state.cumulative_bottom_liquid_exchange_m3
        ),
        cumulative_bottom_gas_exchange_kg=(
            first.state.cumulative_bottom_gas_exchange_kg
        ),
    )
    second = advance_vertical_twofluid(
        restarted,
        parameters,
        dt=dt,
        tee_transaction=transaction,
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )

    swept_area = -imposed_flow * dt / parameters.cell_length_m
    assert first.upper_free_surface_retreat is not None
    assert second.upper_free_surface_retreat is not None
    assert first.upper_free_surface_retreat.interface_cell == 1
    assert second.upper_free_surface_retreat.interface_cell == 1
    assert second.gas_mass_flux_faces_kg_s[1] == 0.0
    assert second.state.Al[1] == pytest.approx(
        initial.Al[1] - 2.0 * swept_area, abs=3.0e-18
    )
    assert second.state.Mg[1] > first.state.Mg[1] > 0.0
    assert math.fsum(second.state.Mg) == pytest.approx(
        math.fsum(initial.Mg), abs=3.0e-18
    )
    assert second.budget.total_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=4.0e-17
    )


def test_upper_retreat_exact_event_projects_flux_and_closes_volume_bitwise() -> None:
    parameters = _parameters(
        cells=4,
        dz=0.10,
        diameter=0.041,
        gravity=0.0,
    )
    area = parameters.full_area_m2
    density = parameters.atmospheric_gas_density_kg_m3
    cut_liquid_area = 0.70 * area
    cut_gas_area = area - cut_liquid_area
    state = VerticalTwoFluidState.from_iterables(
        Al=[area, cut_liquid_area, 0.0, 0.0],
        Ql=[0.0] * 4,
        Mg=[
            0.0,
            density * cut_gas_area * parameters.cell_length_m,
            density * area * parameters.cell_length_m,
            density * area * parameters.cell_length_m,
        ],
        Jg=[0.0] * 4,
    )
    raw_flow = -area * 0.017
    available = cut_liquid_area * parameters.cell_length_m
    dt = available / abs(raw_flow)
    assert abs(raw_flow) * dt != available
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=raw_flow,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    result = advance_vertical_twofluid(
        state,
        parameters,
        dt=dt,
        tee_transaction=transaction,
        top_boundary=AtmosphericTopBoundary(allow_gas_inflow=False),
    )

    projected_flow = result.liquid_volume_flux_faces_m3_s[0]
    assert projected_flow != raw_flow
    assert -projected_flow * dt == available
    assert result.state.Al[1] == 0.0
    assert result.upper_free_surface_retreat is not None
    assert result.upper_free_surface_retreat.liquid_volume_residual_m3 == 0.0
    assert result.budget.liquid_volume_residual_m3 == 0.0


def test_nonzero_cut_gas_cannot_cross_its_lower_liquid_face() -> None:
    parameters = _parameters(
        cells=4, dz=0.10, diameter=0.041, gravity=0.0
    )
    rest = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    donor_velocity = -0.40
    initial = VerticalTwoFluidState.from_iterables(
        Al=rest.Al,
        Ql=rest.Ql,
        Mg=rest.Mg,
        Jg=[0.0, 0.0, rest.Mg[2] * donor_velocity, rest.Mg[3] * donor_velocity],
    )
    base_faces = isothermal_common_pressure_faces(initial, parameters)
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=-1.0e-7,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=(base_faces[0] - base_faces[2]),
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    top = AtmosphericTopBoundary(allow_gas_inflow=False)

    first = advance_vertical_twofluid(
        initial,
        parameters,
        dt=1.0e-4,
        tee_transaction=transaction,
        top_boundary=top,
    )
    assert first.state.Mg[1] > 0.0
    assert first.state.Jg[1] < 0.0
    unprojected_lower_flux = (
        0.5 * first.state.Jg[1] / first.state.Mg[1]
        * first.state.Mg[1]
        / parameters.cell_length_m
    )
    assert unprojected_lower_flux < 0.0

    second = advance_vertical_twofluid(
        first.state,
        parameters,
        dt=1.0e-4,
        tee_transaction=transaction,
        top_boundary=top,
    )
    assert second.upper_free_surface_retreat is not None
    assert second.upper_free_surface_retreat.interface_cell == 1
    assert second.gas_mass_flux_faces_kg_s[1] == 0.0
    assert second.gas_momentum_flux_faces_N[1] == 0.0
    assert second.gas_mass_flux_faces_kg_s[2] < 0.0


def test_retreat_rejects_an_insufficient_top_donor_atomically() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    area = parameters.full_area_m2
    donor_mass = 3.0 * parameters.mass_tolerance_kg
    atmospheric_mass = (
        parameters.atmospheric_gas_density_kg_m3
        * area
        * parameters.cell_length_m
    )
    state = VerticalTwoFluidState.from_iterables(
        Al=[area, area, 0.0, 0.0],
        Ql=[0.0] * 4,
        Mg=[0.0, 0.0, donor_mass, atmospheric_mass],
        Jg=[0.0] * 4,
    )
    snapshot = (state.Al, state.Ql, state.Mg, state.Jg)
    dt = 1.0e-4
    swept_volume = 0.5 * area * parameters.cell_length_m
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=-swept_volume / dt,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    with pytest.raises(TeeTransactionRejected, match="top-gas donor"):
        advance_vertical_twofluid(
            state, parameters, dt=dt, tee_transaction=transaction
        )

    assert (state.Al, state.Ql, state.Mg, state.Jg) == snapshot


def test_retreat_rejects_opposite_and_double_interface_topologies() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    area = parameters.full_area_m2
    gas_mass = (
        parameters.atmospheric_gas_density_kg_m3
        * area
        * parameters.cell_length_m
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=-1.0e-7,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    opposite = VerticalTwoFluidState.from_iterables(
        Al=[0.0, area, area, 0.0],
        Ql=[0.0] * 4,
        Mg=[gas_mass, 0.0, 0.0, gas_mass],
        Jg=[0.0] * 4,
    )
    disconnected = VerticalTwoFluidState.from_iterables(
        Al=[area, 0.0, area, 0.0],
        Ql=[0.0] * 4,
        Mg=[0.0, gas_mass, 0.0, gas_mass],
        Jg=[0.0] * 4,
    )

    with pytest.raises(TeeTransactionRejected, match="orientation"):
        advance_vertical_twofluid(
            opposite, parameters, dt=1.0e-4, tee_transaction=transaction
        )
    with pytest.raises(TeeTransactionRejected, match="not one monotone"):
        advance_vertical_twofluid(
            disconnected, parameters, dt=1.0e-4, tee_transaction=transaction
        )


def test_retreat_interface_cfl_rejects_crossing_multiple_cells() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    dt = 1.0e-4
    more_than_one_cell = 1.01 * (
        parameters.full_area_m2 * parameters.cell_length_m
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=-more_than_one_cell / dt,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )

    with pytest.raises(TeeTransactionRejected, match="interface CFL"):
        advance_vertical_twofluid(
            state, parameters, dt=dt, tee_transaction=transaction
        )


def test_advance_interface_cfl_rejects_crossing_multiple_cells() -> None:
    parameters = _parameters(cells=4, dz=0.10, diameter=0.041)
    state = hydrostatic_column_state(parameters, liquid_height_m=0.20)
    dt = 1.0e-4
    more_than_one_cell = 1.01 * (
        parameters.full_area_m2 * parameters.cell_length_m
    )
    transaction = _complete_tee_transaction(
        parameters,
        west_liquid_flow_m3_s=more_than_one_cell / dt,
        east_liquid_flow_m3_s=0.0,
        gas_mass_flow_to_riser_kg_s=0.0,
        gas_normal_momentum_flow_N=0.0,
        liquid_node_gauge_pressure_Pa=0.0,
        gas_interface_pressure_abs_Pa=parameters.atmospheric_pressure_Pa,
    )
    snapshot = (state.Al, state.Ql, state.Mg, state.Jg, state.time_s)

    with pytest.raises(TeeTransactionRejected, match="interface CFL"):
        advance_vertical_twofluid(
            state, parameters, dt=dt, tee_transaction=transaction
        )

    assert (state.Al, state.Ql, state.Mg, state.Jg, state.time_s) == snapshot
