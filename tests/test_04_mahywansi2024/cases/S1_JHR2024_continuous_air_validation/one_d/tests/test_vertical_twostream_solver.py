import math

import pytest

from model.errors import MissingPhysicalClosure
from model.state import VerticalState
from model.vertical_case1_adapter import ATMOSPHERIC_PRESSURE_PA
from model.vertical_twostream_solver import (
    RiserNodeFluxPacket,
    S1_GRAVITY_M_S2,
    S1_LIQUID_DENSITY_KG_M3,
    S1VerticalClosures,
    S1VerticalTwoStreamSolver,
)


def _full_liquid_state(
    solver: S1VerticalTwoStreamSolver, *, upward_rate_m3_s: float = 0.0
) -> VerticalState:
    n = solver.cell_count
    return VerticalState(
        Aup=(solver.pipe_area_m2,) * n,
        Qup=(upward_rate_m3_s,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(0.0,) * n,
        Jg=(0.0,) * n,
    )


def _full_column_hydrostatic_packet(
    solver: S1VerticalTwoStreamSolver, *, upward_rate_m3_s: float = 0.0
) -> RiserNodeFluxPacket:
    bottom_pressure = (
        ATMOSPHERIC_PRESSURE_PA
        + S1_LIQUID_DENSITY_KG_M3
        * S1_GRAVITY_M_S2
        * 1.02
    )
    speed = (
        0.0
        if upward_rate_m3_s == 0.0
        else upward_rate_m3_s / solver.pipe_area_m2
    )
    return RiserNodeFluxPacket(
        bottom_pressure_pa=bottom_pressure,
        liquid_filled_cell_pressure_pa=solver.hydrostatic_filled_cell_pressures(
            bottom_pressure_pa=bottom_pressure
        ),
        liquid_upward_rate_m3_s=upward_rate_m3_s,
        liquid_upward_speed_m_s=speed,
    )


def test_missing_unpublished_liquid_closures_fail_closed() -> None:
    solver = S1VerticalTwoStreamSolver(cell_count=16)
    assert solver.production_ready is False
    with pytest.raises(MissingPhysicalClosure, match="paper does not publish"):
        solver.advance(
            solver.initial_state,
            dt_s=1.0e-4,
            bottom=solver.source_initial_pressure_packet(),
        )


def test_source_geometry_and_persistent_state_are_not_net_reconstructed() -> None:
    solver = S1VerticalTwoStreamSolver(
        cell_count=40,
        closures=S1VerticalClosures.structural_zero_for_tests(),
    )
    state = solver.initial_state
    assert solver.adapter.diameter_m == pytest.approx(0.0254)
    assert solver.adapter.z_bottom_m == pytest.approx(0.0)
    assert solver.adapter.z_top_m == pytest.approx(1.02)
    assert solver.adapter.initial_water_level_m == pytest.approx(0.5842)
    assert sum((up + down) * solver.cell_length_m for up, down in zip(
        state.Aup, state.Adown, strict=True
    )) == pytest.approx(solver.pipe_area_m2 * 0.5842)
    assert state.net_liquid_discharge == pytest.approx((0.0,) * solver.cell_count)
    assert state.Qup is not state.Qdown


def test_full_liquid_hydrostatic_column_holds_to_roundoff() -> None:
    solver = S1VerticalTwoStreamSolver(
        cell_count=16,
        closures=S1VerticalClosures.structural_zero_for_tests(),
    )
    state = _full_liquid_state(solver)
    result = solver.advance(
        state,
        dt_s=1.0e-3,
        bottom=_full_column_hydrostatic_packet(solver),
    )
    assert result.state.Aup == pytest.approx(state.Aup, abs=1.0e-16)
    assert result.state.Qup == pytest.approx(state.Qup, abs=1.0e-16)
    assert result.state.Adown == pytest.approx(state.Adown, abs=1.0e-16)
    assert result.state.Qdown == pytest.approx(state.Qdown, abs=1.0e-16)
    assert result.state.Mg == pytest.approx(state.Mg, abs=1.0e-16)
    assert result.state.Jg == pytest.approx(state.Jg, abs=1.0e-16)
    assert result.liquid_volume_residual_m3 == pytest.approx(0.0, abs=1.0e-15)
    assert result.gas.mass_residual_kg == pytest.approx(0.0, abs=1.0e-15)
    assert result.mixture_momentum_residual_kg_m_s == pytest.approx(0.0, abs=1.0e-13)
    assert all(math.isfinite(value) for value in result.common_pressure_faces_pa)


def test_gross_bidirectional_streams_and_three_body_recoil_remain_independent() -> None:
    solver = S1VerticalTwoStreamSolver(
        cell_count=8,
        closures=S1VerticalClosures.structural_zero_for_tests(),
    )
    area = solver.pipe_area_m2
    n = solver.cell_count
    rho_air = ATMOSPHERIC_PRESSURE_PA / (287.05 * 293.15)
    q_up = 0.30 * area * 0.20
    q_down = 0.20 * area * 0.15
    state = VerticalState(
        Aup=(0.30 * area,) * n,
        Qup=(q_up,) * n,
        Adown=(0.20 * area,) * n,
        Qdown=(q_down,) * n,
        Mg=(rho_air * 0.50 * area,) * n,
        Jg=(0.0,) * n,
    )
    packet = RiserNodeFluxPacket(
        bottom_pressure_pa=ATMOSPHERIC_PRESSURE_PA,
        liquid_upward_rate_m3_s=q_up,
        liquid_upward_speed_m_s=q_up / (0.30 * area),
        liquid_downward_rate_m3_s=q_down,
        liquid_downward_speed_m_s=q_down / (0.20 * area),
    )
    result = solver.advance(state, dt_s=1.0e-6, bottom=packet)

    assert all(value > 0.0 for value in result.state.Qup)
    assert all(value > 0.0 for value in result.state.Qdown)
    assert any(
        abs(up - down) < max(up, down)
        for up, down in zip(result.state.Qup, result.state.Qdown, strict=True)
    )
    assert result.state.Qup != result.state.Qdown
    assert result.three_body_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-12
    )
    assert result.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-15)
    assert result.gas.mass_residual_kg == pytest.approx(0.0, abs=2.0e-15)
    assert result.gas.momentum_residual_kg_m_s == pytest.approx(0.0, abs=2.0e-15)
    assert result.mixture_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=5.0e-11
    )


def test_explicit_bottom_gas_flux_packet_enters_conservatively() -> None:
    solver = S1VerticalTwoStreamSolver(
        cell_count=8,
        closures=S1VerticalClosures.structural_zero_for_tests(),
    )
    n = solver.cell_count
    rho_air = ATMOSPHERIC_PRESSURE_PA / (287.05 * 293.15)
    state = VerticalState(
        Aup=(0.0,) * n,
        Qup=(0.0,) * n,
        Adown=(0.0,) * n,
        Qdown=(0.0,) * n,
        Mg=(rho_air * solver.pipe_area_m2,) * n,
        Jg=(0.0,) * n,
    )
    gas_rate = 2.0e-7
    dt = 1.0e-5
    result = solver.advance(
        state,
        dt_s=dt,
        bottom=RiserNodeFluxPacket(
            bottom_pressure_pa=ATMOSPHERIC_PRESSURE_PA,
            gas_upward_mass_rate_kg_s=gas_rate,
            gas_upward_speed_m_s=0.10,
        ),
    )
    assert result.gas.final_mass_kg - result.gas.initial_mass_kg == pytest.approx(
        gas_rate * dt, rel=1.0e-11, abs=1.0e-16
    )
    assert result.gas.bottom_net_mass_rate_kg_s == pytest.approx(gas_rate)
    assert result.gas.mass_residual_kg == pytest.approx(0.0, abs=2.0e-15)
    assert all(math.isfinite(value) and value > 0.0 for value in result.gas_pressure_cells_pa)
    assert result.gas_pressure_cells_pa[0] > ATMOSPHERIC_PRESSURE_PA
    assert result.mixture_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=5.0e-11
    )


def test_top_donor_outflow_integrates_for_0p02s_without_nan() -> None:
    solver = S1VerticalTwoStreamSolver(
        cell_count=16,
        closures=S1VerticalClosures.structural_zero_for_tests(),
    )
    q = 0.05 * solver.pipe_area_m2
    state = _full_liquid_state(solver, upward_rate_m3_s=q)
    packet = _full_column_hydrostatic_packet(solver, upward_rate_m3_s=q)
    dt = 1.0e-3
    integrated_top_volume = 0.0
    for _ in range(20):
        result = solver.advance(state, dt_s=dt, bottom=packet)
        state = result.state
        integrated_top_volume += result.top_liquid_outflow_volume_m3
        vectors = (
            state.Aup,
            state.Qup,
            state.Adown,
            state.Qdown,
            state.Mg,
            state.Jg,
            result.common_pressure_faces_pa,
        )
        assert all(math.isfinite(value) for vector in vectors for value in vector)
        assert result.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-15)
        assert result.gas.mass_residual_kg == pytest.approx(0.0, abs=2.0e-15)
        assert result.mixture_momentum_residual_kg_m_s == pytest.approx(
            0.0, abs=5.0e-11
        )

    assert integrated_top_volume == pytest.approx(q * 0.02, rel=1.0e-12)
    assert result.top_liquid_outflow_rate_m3_s == pytest.approx(q, rel=1.0e-12)
    assert result.validation_only is True
    assert result.production_ready is False
