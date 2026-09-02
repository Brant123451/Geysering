import math

import pytest

from model.errors import ContractViolation
from model.flux import SupplyBranchDelta
from model.state import SupplyBranchState
from model.supply_branch_twophase import (
    CLOSURE_PROVENANCE,
    PUBLISHED_GAS_GAUGE_PRESSURE_PA,
    SUPPLY_BRANCH_ALLOWED_CELL_COUNTS,
    SUPPLY_BRANCH_CELL_COUNT,
    SUPPLY_BRANCH_LENGTH_M,
    SupplyBottomNodeCondition,
    SupplyBranchGeometry,
    SupplyBranchTwoPhaseSolver,
    f0_supply_smooth_pipe_darcy_factor,
)


def _solver(cell_count: int = SUPPLY_BRANCH_CELL_COUNT) -> SupplyBranchTwoPhaseSolver:
    return SupplyBranchTwoPhaseSolver(
        geometry=SupplyBranchGeometry(cell_count=cell_count)
    )


def _assert_finite_state(state: SupplyBranchState) -> None:
    assert all(
        math.isfinite(value)
        for field in (state.Al, state.Ql, state.Mg, state.Jg)
        for value in field
    )


def test_geometry_provenance_and_initial_state_are_source_aligned_full_water() -> None:
    solver = _solver()
    state = solver.initial_state()

    assert solver.geometry.length_m == pytest.approx(SUPPLY_BRANCH_LENGTH_M)
    assert state.cell_count == solver.geometry.cell_count == SUPPLY_BRANCH_CELL_COUNT == 14
    assert SUPPLY_BRANCH_ALLOWED_CELL_COUNTS == (14, 28)
    assert state.Al == pytest.approx((solver.geometry.area_m2,) * 14)
    assert state.Ql == (0.0,) * 14
    assert state.Mg == (0.0,) * 14
    assert state.Jg == (0.0,) * 14
    assert solver.pressure_reservoir.reservoir_absolute_pressure_Pa == pytest.approx(
        solver.config.atmospheric_pressure_Pa + PUBLISHED_GAS_GAUGE_PRESSURE_PA
    )
    assert solver.config.stage2_top_alpha_water == 0.0
    assert solver.stage1_top_boundary == "impermeable_wall"
    assert solver.stage2_top_boundary == "5700Pa_gauge_pure_air_pressure_Riemann"
    assert CLOSURE_PROVENANCE.endswith("not_published_not_tuned")
    assert not solver.alignment_ready
    assert not solver.production_ready


def test_strict_phase_pairing_rejects_vacuum_area_and_zero_area_gas_mass() -> None:
    solver = _solver()
    area = solver.geometry.area_m2
    vacuum = SupplyBranchState(
        Al=(area,) * 13 + (0.5 * area,),
        Ql=(0.0,) * 14,
        Mg=(0.0,) * 14,
        Jg=(0.0,) * 14,
    )
    gas_without_area = SupplyBranchState(
        Al=(area,) * 14,
        Ql=(0.0,) * 14,
        Mg=(0.0,) * 13 + (1.0e-5,),
        Jg=(0.0,) * 14,
    )

    with pytest.raises(ContractViolation, match="Ag>0 iff"):
        solver.validate_state(vacuum)
    with pytest.raises(ContractViolation, match="Ag>0 iff"):
        solver.validate_state(gas_without_area)


def test_stage1_top_wall_preserves_full_water_hydrostatic_equilibrium() -> None:
    solver = _solver()
    initial = solver.initial_state()
    dt = 0.8 * solver.stable_timestep_s(initial)
    result = solver.propose_atomic_step(
        initial, dt, stage="stage1_closed", transaction_id="stage1-static"
    )

    assert result.state == initial
    assert result.delta == SupplyBranchDelta.zeros(14)
    assert result.bottom.base_state_token
    assert result.bottom.bottom_momentum_flux_upward_N == pytest.approx(
        result.bottom.bottom_absolute_pressure_Pa * solver.geometry.area_m2
    )
    assert result.bottom.liquid_net_into_branch_m3_s == 0.0
    assert result.bottom.gas_net_into_branch_kg_s == 0.0
    assert result.ledger.top_gas_net_into_branch_kg == 0.0
    assert result.ledger.liquid_volume_residual_m3 == 0.0
    assert result.ledger.gas_mass_residual_kg == 0.0
    assert result.ledger.phase_volume_residual_m3 == 0.0
    assert result.ledger.mixture_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=5.0e-18
    )
    assert result.ledger.wall_momentum_impulse_kg_m_s == 0.0
    assert result.ledger.liquid_darcy_factor == 0.0
    assert result.ledger.gas_darcy_factor == 0.0


def test_f0_supply_darcy_law_has_frozen_zero_laminar_transition_and_turbulent_branches() -> None:
    assert f0_supply_smooth_pipe_darcy_factor(0.0) == 0.0
    assert f0_supply_smooth_pipe_darcy_factor(1000.0) == pytest.approx(0.064)
    re = 3000.0
    weight = (re - 2300.0) / 1700.0
    expected = (1.0 - weight) * 64.0 / re + weight * 0.3164 / re**0.25
    assert f0_supply_smooth_pipe_darcy_factor(re) == pytest.approx(expected)
    assert f0_supply_smooth_pipe_darcy_factor(10000.0) == pytest.approx(
        0.3164 / 10000.0**0.25
    )


@pytest.mark.parametrize("velocity", (0.20, -0.20))
def test_full_liquid_wall_shear_is_sign_preserving_and_matches_frozen_formula(
    velocity: float,
) -> None:
    solver = _solver()
    state = solver.state_from_bulk(
        gas_volume_m3=0.0,
        gas_mass_kg=0.0,
        liquid_velocity_upward_m_s=velocity,
    )
    dt = 0.01
    relaxed, audit = solver._apply_f0_wall_shear(state, dt)
    before = solver.inventory(state)
    after = solver.inventory(relaxed)
    reynolds = (
        solver.config.liquid_density_kg_m3
        * abs(velocity)
        * solver.geometry.diameter_m
        / solver.config.liquid_viscosity_Pa_s
    )
    factor = f0_supply_smooth_pipe_darcy_factor(reynolds)
    expected_velocity = velocity / (
        1.0 + factor * abs(velocity) * dt / (2.0 * solver.geometry.diameter_m)
    )
    observed_velocity = after.liquid_momentum_kg_m_s / (
        solver.config.liquid_density_kg_m3 * after.liquid_volume_m3
    )
    assert observed_velocity == pytest.approx(expected_velocity)
    assert math.copysign(1.0, observed_velocity) == math.copysign(1.0, velocity)
    assert abs(observed_velocity) < abs(velocity)
    assert audit.liquid_darcy_factor == pytest.approx(factor)
    assert audit.gas_darcy_factor == 0.0
    assert audit.total_wall_impulse_kg_m_s == pytest.approx(
        after.mixture_momentum_kg_m_s - before.mixture_momentum_kg_m_s
    )
    assert audit.total_wall_impulse_kg_m_s * velocity < 0.0


@pytest.mark.parametrize("velocity", (0.30, -0.30))
def test_full_gas_wall_shear_is_sign_preserving_and_uses_frozen_gas_viscosity(
    velocity: float,
) -> None:
    solver = _solver()
    volume = solver.geometry.total_volume_m3
    pressure = 107025.0
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
        gas_velocity_upward_m_s=velocity,
    )
    dt = 0.01
    relaxed, audit = solver._apply_f0_wall_shear(state, dt)
    before = solver.inventory(state)
    after = solver.inventory(relaxed)
    density = before.gas_mass_kg / before.gas_volume_m3
    reynolds = (
        density
        * abs(velocity)
        * solver.geometry.diameter_m
        / solver.config.gas_viscosity_Pa_s
    )
    factor = f0_supply_smooth_pipe_darcy_factor(reynolds)
    expected_velocity = velocity / (
        1.0 + factor * abs(velocity) * dt / (2.0 * solver.geometry.diameter_m)
    )
    observed_velocity = after.gas_momentum_kg_m_s / after.gas_mass_kg
    assert observed_velocity == pytest.approx(expected_velocity)
    assert math.copysign(1.0, observed_velocity) == math.copysign(1.0, velocity)
    assert abs(observed_velocity) < abs(velocity)
    assert audit.gas_darcy_factor == pytest.approx(factor)
    assert audit.liquid_darcy_factor == 0.0
    assert audit.total_wall_impulse_kg_m_s == pytest.approx(
        after.mixture_momentum_kg_m_s - before.mixture_momentum_kg_m_s
    )
    assert audit.total_wall_impulse_kg_m_s * velocity < 0.0


def test_mixed_plug_wall_impulses_close_without_breaking_common_contact_velocity() -> None:
    solver = _solver()
    volume = 0.35 * solver.geometry.total_volume_m3
    pressure = 106000.0
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
        liquid_velocity_upward_m_s=-0.15,
        gas_velocity_upward_m_s=-0.15,
    )
    relaxed, audit = solver._apply_f0_wall_shear(state, 0.01)
    before = solver.inventory(state)
    after = solver.inventory(relaxed)
    ul = after.liquid_momentum_kg_m_s / (
        solver.config.liquid_density_kg_m3 * after.liquid_volume_m3
    )
    ug = after.gas_momentum_kg_m_s / after.gas_mass_kg
    assert ul == pytest.approx(ug)
    assert -0.15 < ul < 0.0
    assert audit.liquid_darcy_factor > 0.0
    assert audit.gas_darcy_factor > 0.0
    assert audit.total_wall_impulse_kg_m_s == pytest.approx(
        audit.liquid_wall_impulse_kg_m_s + audit.gas_wall_impulse_kg_m_s
    )
    assert audit.total_wall_impulse_kg_m_s == pytest.approx(
        after.mixture_momentum_kg_m_s - before.mixture_momentum_kg_m_s
    )


def test_wall_failure_rolls_back_pure_supply_proposal(monkeypatch) -> None:
    solver = _solver()
    state = solver.initial_state()
    before = state

    def reject_wall(*_args, **_kwargs):
        raise ContractViolation("manufactured wall audit rejection")

    monkeypatch.setattr(solver, "_apply_f0_wall_shear", reject_wall)
    with pytest.raises(ContractViolation, match="wall audit rejection"):
        solver.propose_atomic_step(
            state,
            0.5 * solver.stable_timestep_s(state),
            stage="stage2_pressure_reservoir",
            transaction_id="wall-rollback",
        )
    assert state == before


def test_first_stage2_air_entry_creates_volume_and_simultaneously_ejects_water() -> None:
    solver = _solver()
    initial = solver.initial_state()
    dt = 0.5 * solver.stable_timestep_s(initial)
    result = solver.propose_atomic_step(
        initial, dt, stage="stage2_pressure_reservoir", transaction_id="first-air"
    )
    before = result.ledger.before
    after = result.ledger.after

    assert after.gas_mass_kg > 0.0
    assert after.gas_volume_m3 > 0.0
    assert result.ledger.top_gas_net_into_branch_kg > 0.0
    assert result.bottom.liquid_downward_rate_m3_s > 0.0
    assert result.bottom.liquid_upward_rate_m3_s == 0.0
    assert math.isfinite(result.bottom.bottom_momentum_flux_upward_N)
    displaced = before.liquid_volume_m3 - after.liquid_volume_m3
    assert displaced == pytest.approx(after.gas_volume_m3, rel=0.0, abs=2.0e-18)
    assert displaced == pytest.approx(
        result.bottom.liquid_downward_rate_m3_s * dt,
        rel=0.0,
        abs=2.0e-18,
    )
    assert result.ledger.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=2.0e-18)
    assert result.ledger.acoustic_projection_impulse_kg_m_s == pytest.approx(
        0.0, abs=1.0e-18
    )
    assert result.ledger.wall_momentum_impulse_kg_m_s == pytest.approx(
        result.ledger.liquid_wall_impulse_kg_m_s
        + result.ledger.gas_wall_impulse_kg_m_s
    )
    assert result.ledger.wall_momentum_impulse_kg_m_s > 0.0
    assert result.ledger.liquid_darcy_factor > 0.0
    assert result.ledger.gas_darcy_factor > 0.0
    # Gas appears only in cells whose complementary area was created in the
    # same atomic proposal; no Mg was written into a still-full-water cell.
    for al, mg in zip(result.state.Al, result.state.Mg, strict=True):
        assert ((solver.geometry.area_m2 - al) > 0.0) == (mg > 0.0)


def test_first_air_rate_roundtrip_keeps_sub_packing_tolerance_Ag_Mg_pair() -> None:
    solver = _solver()
    initial = solver.initial_state()
    dt = 1.0e-7
    result = solver.propose_atomic_step(
        initial, dt, stage="stage2_pressure_reservoir", transaction_id="tiny-first-air"
    )
    rate = solver._rate_delta(result.delta, dt)

    def add(values, derivatives):
        return tuple(
            value + dt * derivative
            for value, derivative in zip(values, derivatives, strict=True)
        )

    candidate = SupplyBranchState(
        Al=add(initial.Al, rate.Al),
        Ql=add(initial.Ql, rate.Ql),
        Mg=add(initial.Mg, rate.Mg),
        Jg=add(initial.Jg, rate.Jg),
    )
    gas_area = solver.geometry.area_m2 - candidate.Al[-1]
    assert 0.0 < gas_area < 1.0e-12
    assert candidate.Mg[-1] > 0.0
    solver.validate_state(candidate)
    assert solver.inventory(candidate).gas_mass_kg == pytest.approx(
        result.ledger.after.gas_mass_kg, rel=0.0, abs=2.0e-24
    )


def test_newborn_gas_implicit_HLL_respects_finite_mass_capacity() -> None:
    solver = _solver()
    dt = 1.0e-7
    volume = 5.35e-17
    pressure = solver.pressure_reservoir.reservoir_absolute_pressure_Pa + 0.36
    velocity = -1.06e-6
    mass = pressure / solver.config.rt_J_kg * volume
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=mass,
        liquid_velocity_upward_m_s=velocity,
        gas_velocity_upward_m_s=velocity,
    )
    explicit = solver.pressure_reservoir.evaluate(
        node_absolute_pressure_Pa=pressure,
        node_axial_velocity_m_s=-velocity,
        inlet_area_m2=solver.geometry.area_m2,
    )
    assert mass + dt * explicit.mass_flow_kg_s < 0.0

    result = solver.propose_atomic_step(
        state,
        dt,
        stage="stage2_pressure_reservoir",
        transaction_id="implicit-finite-newborn-gas",
    )

    assert result.ledger.after.gas_mass_kg > 0.0
    assert result.ledger.after.gas_volume_m3 > 0.0
    assert result.ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=2.0e-22)
    assert result.ledger.liquid_volume_residual_m3 == pytest.approx(
        0.0, abs=2.0e-20
    )
    assert result.ledger.mixture_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=2.0e-15
    )


def test_downward_first_air_piston_CFL_limits_liquid_not_growing_gas() -> None:
    solver = _solver()
    volume = 5.35e-17
    velocity = -1.06e-6
    pressure = solver.pressure_reservoir.reservoir_absolute_pressure_Pa
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
        liquid_velocity_upward_m_s=velocity,
        gas_velocity_upward_m_s=velocity,
    )
    false_gas_exhaustion_limit = volume / (
        solver.geometry.area_m2 * abs(velocity)
    )
    stable = solver.stable_timestep_s(state)

    assert stable > 10.0 * false_gas_exhaustion_limit
    assert stable <= (
        solver.inventory(state).liquid_volume_m3
        / (solver.geometry.area_m2 * abs(velocity))
    )


def test_first_air_proposal_is_repeatable_and_failure_rolls_back(monkeypatch) -> None:
    solver = _solver()
    initial = solver.initial_state()
    dt = 1.0e-7
    before = repr(initial)
    first = solver.propose_atomic_step(
        initial, dt, stage="stage2_pressure_reservoir", transaction_id="repeat-first"
    )
    second = solver.propose_atomic_step(
        initial, dt, stage="stage2_pressure_reservoir", transaction_id="repeat-first"
    )
    assert first == second
    assert repr(initial) == before

    newborn = first.state
    newborn_before = repr(newborn)

    def reject_finite_inventory(**_kwargs):
        raise ContractViolation("manufactured implicit finite-inventory rejection")

    monkeypatch.setattr(
        solver, "_implicit_stage2_top_gas_exchange", reject_finite_inventory
    )
    with pytest.raises(ContractViolation, match="finite-inventory rejection"):
        solver.propose_atomic_step(
            newborn,
            dt,
            stage="stage2_pressure_reservoir",
            transaction_id="rollback-newborn",
        )
    assert repr(newborn) == newborn_before


def test_closed_mixed_column_conserves_both_phases_and_momentum() -> None:
    solver = _solver()
    volume = 0.35 * solver.geometry.total_volume_m3
    pressure = 105800.0
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
    )
    bottom = SupplyBottomNodeCondition(
        absolute_pressure_Pa=solver.source_hydrostatic_bottom_pressure_Pa,
        wall=True,
    )
    result = solver.propose_atomic_step(
        state,
        0.5 * solver.stable_timestep_s(state),
        stage="stage1_closed",
        bottom=bottom,
        transaction_id="closed-mixed",
    )

    assert result.state == state
    assert result.ledger.after == result.ledger.before
    assert result.ledger.top_gas_net_into_branch_kg == 0.0
    assert result.ledger.bottom_liquid_net_into_branch_m3 == 0.0
    assert result.ledger.bottom_gas_net_into_branch_kg == 0.0
    assert result.ledger.mixture_momentum_residual_kg_m_s == pytest.approx(
        0.0, abs=5.0e-18
    )


def test_liquid_and_reservoir_backflow_are_signed_without_clipping() -> None:
    solver = _solver()
    volume = 0.25 * solver.geometry.total_volume_m3
    pressure = 1.08 * solver.pressure_reservoir.reservoir_absolute_pressure_Pa
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
        liquid_velocity_upward_m_s=0.02,
    )
    dt = 0.25 * solver.stable_timestep_s(state)
    result = solver.propose_atomic_step(
        state, dt, stage="stage2_pressure_reservoir", transaction_id="backflow"
    )

    assert result.bottom.liquid_upward_rate_m3_s > 0.0
    assert result.bottom.liquid_downward_rate_m3_s == 0.0
    assert result.ledger.top_gas_net_into_branch_kg < 0.0
    assert result.ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=2.0e-16)
    assert result.ledger.liquid_volume_residual_m3 == pytest.approx(0.0, abs=2.0e-16)


def test_full_gas_state_exports_explicit_bottom_gas_gross_packet() -> None:
    solver = _solver()
    volume = solver.geometry.total_volume_m3
    pressure = 120000.0
    state = solver.state_from_bulk(
        gas_volume_m3=volume,
        gas_mass_kg=pressure / solver.config.rt_J_kg * volume,
    )
    result = solver.propose_atomic_step(
        state,
        0.2 * solver.stable_timestep_s(state),
        stage="stage2_pressure_reservoir",
        transaction_id="gas-to-node",
    )

    assert result.bottom.gas_downward_mass_rate_kg_s > 0.0
    assert result.bottom.gas_upward_mass_rate_kg_s == 0.0
    assert result.bottom.gas_downward_speed_m_s > 0.0
    assert result.bottom.liquid_upward_rate_m3_s == 0.0
    assert result.bottom.liquid_downward_rate_m3_s == 0.0
    assert result.ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=2.0e-16)


def test_component_delta_is_the_canonical_atomic_supply_delta() -> None:
    solver = _solver()
    state = solver.initial_state()
    result = solver.propose_atomic_step(
        state,
        0.25 * solver.stable_timestep_s(state),
        stage="stage2_pressure_reservoir",
        transaction_id="canonical-delta",
    )

    assert isinstance(result.delta, SupplyBranchDelta)
    for before, delta, after in zip(
        state.Al, result.delta.Al, result.state.Al, strict=True
    ):
        assert before + delta == pytest.approx(after, abs=1.0e-20)
    for before, delta, after in zip(
        state.Mg, result.delta.Mg, result.state.Mg, strict=True
    ):
        assert before + delta == pytest.approx(after, abs=1.0e-20)


def test_cfl_and_positivity_fail_closed_without_material_clipping() -> None:
    solver = _solver()
    state = solver.initial_state()
    with pytest.raises(ContractViolation, match="exceeds CFL"):
        solver.propose_atomic_step(
            state,
            1.01 * solver.stable_timestep_s(state),
            stage="stage2_pressure_reservoir",
        )


def test_supply_grid_rejects_any_level_outside_the_frozen_pair() -> None:
    for cells in (13, 15, 27, 29, 56):
        with pytest.raises(ContractViolation, match="exactly 14 or 28"):
            SupplyBranchGeometry(cell_count=cells)


@pytest.mark.parametrize("cell_count", SUPPLY_BRANCH_ALLOWED_CELL_COUNTS)
def test_stage2_point_zero_two_second_smoke_is_finite_bounded_and_conservative(
    cell_count: int,
) -> None:
    solver = _solver(cell_count)
    initial = solver.initial_state()
    assert initial.cell_count == cell_count
    assert initial.Al == pytest.approx((solver.geometry.area_m2,) * cell_count)
    assert initial.Ql == (0.0,) * cell_count
    assert initial.Mg == (0.0,) * cell_count
    assert initial.Jg == (0.0,) * cell_count
    result = solver.advance(
        initial,
        0.02,
        stage="stage2_pressure_reservoir",
        transaction_prefix="supply-smoke",
    )
    final_inventory = solver.inventory(result.state)

    assert result.ledger
    assert len(result.ledger) == len(result.packets)
    _assert_finite_state(result.state)
    solver.validate_state(result.state)
    assert 0.0 < final_inventory.gas_volume_m3 < solver.geometry.total_volume_m3
    assert 0.0 < final_inventory.liquid_volume_m3 < solver.geometry.total_volume_m3
    assert final_inventory.gas_mass_kg > 0.0
    assert max(abs(entry.liquid_volume_residual_m3) for entry in result.ledger) < 2.0e-16
    assert max(abs(entry.gas_mass_residual_kg) for entry in result.ledger) < 2.0e-16
    assert max(abs(entry.phase_volume_residual_m3) for entry in result.ledger) < 2.0e-16
    assert max(
        abs(entry.mixture_momentum_residual_kg_m_s) for entry in result.ledger
    ) < 2.0e-15
    assert max(
        abs(entry.acoustic_projection_impulse_kg_m_s) for entry in result.ledger
    ) == 0.0
    assert max(entry.maximum_courant for entry in result.ledger) <= solver.config.cfl * (
        1.0 + 1.0e-10
    )
