import math

import pytest

from model.errors import MissingPhysicalClosure
from model.horizontal_case1_adapter import Case1HorizontalLiquidAdapter
from model.horizontal_distributed import (
    AIR_STUB_LENGTH_M,
    AirStubState,
    DistributedHorizontalState,
    HorizontalClosureSet,
    HorizontalDistributedSolver,
)
from model.state import HorizontalState


def _solver(closures=None):
    return HorizontalDistributedSolver(
        Case1HorizontalLiquidAdapter(),
        closures=(
            HorizontalClosureSet.verification_inviscid()
            if closures is None
            else closures
        ),
    )


def test_closure_selection_is_explicit_and_formal_trajectory_fails_closed() -> None:
    with pytest.raises(MissingPhysicalClosure, match="selected explicitly"):
        HorizontalDistributedSolver(Case1HorizontalLiquidAdapter(), closures=None)
    solver = _solver()
    assert not solver.alignment_ready
    assert not solver.source_aligned_trajectory_ready
    with pytest.raises(MissingPhysicalClosure, match="no liquid-intrusion"):
        solver.assert_source_aligned_trajectory_ready()


def test_air_stub_is_persistent_fv_mass_and_momentum_not_a_zero_volume_node() -> None:
    solver = _solver()
    state = solver.initial_state()
    assert solver.air_stub.length_m == pytest.approx(AIR_STUB_LENGTH_M)
    assert AIR_STUB_LENGTH_M == pytest.approx(0.1373)
    assert state.air_stub.cell_count == 14
    assert all(mass > 0.0 for mass in state.air_stub.Mg)
    assert state.air_stub.Jg == pytest.approx((0.0,) * 14)
    assert state.main.Mg == pytest.approx((0.0,) * state.main.cell_count)
    inventory = solver.inventory(state)
    expected_stub_mass = sum(state.air_stub.Mg) * solver.air_stub.dz_m
    assert inventory.total_gas_mass_kg == pytest.approx(expected_stub_mass)


def test_gas_occupied_main_cell_uses_exact_complement_and_isothermal_eos() -> None:
    solver = _solver()
    initial = solver.initial_state()
    index = solver.tee_face - 1
    al = list(initial.main.Al)
    mg = list(initial.main.Mg)
    jg = list(initial.main.Jg)
    al[index] = 0.8 * solver.area
    target_pressure = 106500.0
    gas_area = solver.area - al[index]
    mg[index] = target_pressure * gas_area / solver.config.rt_J_kg
    state = DistributedHorizontalState(
        time_s=0.0,
        main=HorizontalState(al, initial.main.Ql, mg, jg),
        air_stub=initial.air_stub,
    )
    solver.validate_state(state)
    pressure = solver.gas_pressure_Pa(state)
    assert solver.area - state.main.Al[index] == pytest.approx(gas_area)
    assert pressure[index] == pytest.approx(target_pressure)


def test_table1_centreline_head_maps_to_crown_pressure_with_one_radius_offset() -> None:
    solver = _solver()
    state = solver.initial_state()
    pressure = solver._main_pressures(solver._arrays(state))
    expected = 101325.0 + 998.4 * 9.81 * (0.586 - 0.0254 / 2.0)
    wrong_diameter_offset = 101325.0 + 998.4 * 9.81 * (0.586 - 0.0254)
    assert pressure == pytest.approx((expected,) * state.main.cell_count)
    assert expected - wrong_diameter_offset == pytest.approx(
        998.4 * 9.81 * 0.0254 / 2.0
    )


def test_stage1_one_cfl_step_is_finite_and_closed_gas_mass_is_conservative() -> None:
    solver = _solver()
    initial = solver.initial_state()
    dt = solver.stable_timestep_s(initial)
    final, ledger = solver._ssprk2_step(initial, dt, "stage1_closed")
    assert final.time_s == pytest.approx(dt)
    assert ledger.reservoir_gas_exchange_kg == pytest.approx(0.0, abs=1.0e-18)
    assert ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=1.0e-14)
    assert ledger.liquid_volume_residual_m3 == pytest.approx(0.0, abs=1.0e-14)
    assert ledger.node_mass_residual_kg_s < 1.0e-12
    assert ledger.maximum_courant <= solver.config.cfl * (1.0 + 1.0e-12)
    assert all(math.isfinite(value) for value in final.main.Ql)
    assert final.air_stub.cell_count == initial.air_stub.cell_count


def test_stage2_pressure_reservoir_short_source_smoke_closes_gas_ledger() -> None:
    solver = _solver()
    initial = solver.initial_state()
    # A slight, declared synthetic under-pressure exercises inflow through the
    # actual stub top.  It is not a fitted S1 initial condition.
    underpressured = DistributedHorizontalState(
        time_s=0.0,
        main=initial.main,
        air_stub=AirStubState(
            Mg=tuple(0.999 * value for value in initial.air_stub.Mg),
            Jg=initial.air_stub.Jg,
        ),
    )
    dt = 0.25 * solver.stable_timestep_s(underpressured)
    final, ledger = solver._ssprk2_step(
        underpressured, dt, "stage2_pressure_reservoir"
    )
    assert ledger.reservoir_gas_exchange_kg > 0.0
    assert ledger.after.total_gas_mass_kg > ledger.before.total_gas_mass_kg
    assert ledger.gas_mass_residual_kg == pytest.approx(0.0, abs=1.0e-13)
    assert final.air_stub.Mg[-1] > 0.0


def test_gas_position_observer_and_drag_recoil_are_not_result_prescriptions() -> None:
    closures = HorizontalClosureSet.declared_smooth_pipe_unvalidated(
        interphase_drag_coefficient=0.44
    )
    solver = _solver(closures)
    initial = solver.initial_state()
    al = list(initial.main.Al)
    mg = list(initial.main.Mg)
    jg = list(initial.main.Jg)
    for index, velocity in ((solver.tee_face - 1, -0.5), (solver.tee_face, 0.5)):
        al[index] = 0.75 * solver.area
        gas_area = solver.area - al[index]
        mg[index] = 106500.0 * gas_area / solver.config.rt_J_kg
        jg[index] = mg[index] * velocity
    state = DistributedHorizontalState(
        0.0,
        HorizontalState(al, initial.main.Ql, mg, jg),
        initial.air_stub,
    )
    observation = solver.gas_positions(state)
    assert observation.gas_cell_count == 2
    assert observation.tail_x_m == pytest.approx(solver.adapter.grid.air_tee_x_m - solver.dx)
    assert observation.nose_x_m == pytest.approx(solver.adapter.grid.air_tee_x_m + solver.dx)
    _, budget = solver._rhs(state, "stage1_closed")
    assert budget.interphase_recoil_residual_N_per_m_integral == pytest.approx(
        0.0, abs=1.0e-15
    )
    assert not solver.alignment_ready
