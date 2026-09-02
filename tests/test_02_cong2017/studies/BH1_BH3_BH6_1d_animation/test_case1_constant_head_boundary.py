from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from case1_constant_head_boundary import Case1ConstantHeadBoundary
from case1_mirrored_horizontal import Campaign2Case1MirroredHorizontal


RESERVOIR_HEAD_M = 0.66


def build_horizontal() -> Campaign2Case1MirroredHorizontal:
    return Campaign2Case1MirroredHorizontal(
        length=6.59,
        diameter=0.05,
        physical_valve_x=5.98,
        physical_riser_x=3.47,
        initial_water_head_from_invert=RESERVOIR_HEAD_M,
        dx=0.02,
        wave_speed=28.0,
        valve_open_time=0.20,
        gas_temperature=296.15,
    )


def set_reservoir_adjacent_uniform_head(solver, state, head_m: float):
    area = np.asarray(state.area, dtype=float).copy()
    discharge = np.asarray(state.discharge, dtype=float).copy()
    area[-2:] = float(solver.section.area_from_head(head_m))
    discharge[-2:] = 0.0
    return replace(state, area=area, discharge=discharge)


def test_boundary_location_and_contract_are_physical_not_case_tuned() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)

    assert boundary.physical_boundary_x_m == 0.0
    assert boundary.mirrored_boundary_x_m == pytest.approx(6.59)
    assert boundary.reservoir_head_from_invert_m == RESERVOIR_HEAD_M
    provenance = boundary.provenance()
    assert provenance["valve_hydraulic_time_coupling"] == (
        "none; physical dt only"
    )
    assert not any("BH1" in str(value) for value in provenance.values())
    assert not any("geyser" in str(value).lower() for value in provenance.values())


def test_exact_hydrostatic_equilibrium_has_strictly_zero_flow() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = set_reservoir_adjacent_uniform_head(
        solver,
        solver.initial_state(),
        RESERVOIR_HEAD_M,
    )

    solution = boundary.solve(state, dt=2.0e-4)

    assert solution.characteristic_foot_velocity_m_s == 0.0
    assert solution.mirrored_boundary_velocity_m_s == 0.0
    assert solution.mirrored_boundary_discharge_m3_s == 0.0
    assert solution.physical_inflow_m3_s == 0.0
    committed = boundary.commit(state, dt=2.0e-4)
    assert committed.liquid_volume_to_horizontal_m3 == 0.0
    assert committed.horizontal_liquid_volume_change_m3 == 0.0
    assert committed.mass_balance_residual_m3 == 0.0


def test_lower_domain_pressure_draws_water_into_physical_domain() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = set_reservoir_adjacent_uniform_head(
        solver,
        solver.initial_state(),
        0.40,
    )

    solution = boundary.solve(state, dt=2.0e-4)

    assert solution.characteristic_foot_head_m == pytest.approx(0.40)
    assert solution.mirrored_boundary_velocity_m_s < 0.0
    assert solution.mirrored_boundary_discharge_m3_s < 0.0
    assert solution.physical_inflow_m3_s > 0.0


def test_higher_domain_pressure_returns_water_to_reservoir() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = set_reservoir_adjacent_uniform_head(
        solver,
        solver.initial_state(),
        0.90,
    )

    solution = boundary.solve(state, dt=2.0e-4)

    assert solution.characteristic_foot_head_m == pytest.approx(0.90)
    assert solution.mirrored_boundary_velocity_m_s > 0.0
    assert solution.mirrored_boundary_discharge_m3_s > 0.0
    assert solution.physical_inflow_m3_s < 0.0


def test_commit_adds_characteristic_volume_and_closes_mass_ledger() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = set_reservoir_adjacent_uniform_head(
        solver,
        solver.initial_state(),
        0.40,
    )
    dt = 2.0e-4
    volume_before = float(np.sum(state.area) * solver.dx)

    first = boundary.commit(state, dt)
    volume_after = float(np.sum(first.state.area) * solver.dx)

    assert first.liquid_volume_to_horizontal_m3 == pytest.approx(
        first.solution.physical_inflow_m3_s * dt,
        rel=0.0,
        abs=2.0e-18,
    )
    assert volume_after - volume_before == pytest.approx(
        first.liquid_volume_to_horizontal_m3,
        rel=0.0,
        abs=2.0e-18,
    )
    assert (
        first.horizontal_liquid_volume_change_m3
        + first.reservoir_liquid_volume_change_m3
    ) == 0.0
    assert first.mass_balance_residual_m3 == 0.0

    # Exercise the opposite sign in the same ledger; no result/case branch is
    # involved and the two one-way counters remain independently auditable.
    high = set_reservoir_adjacent_uniform_head(solver, first.state, 0.90)
    second = boundary.commit(high, dt)
    ledger = boundary.ledger()
    assert second.liquid_volume_to_horizontal_m3 < 0.0
    assert ledger["commit_count"] == 2
    assert ledger["inflow_to_horizontal_m3"] > 0.0
    assert ledger["outflow_from_horizontal_m3"] > 0.0
    assert ledger["mass_balance_residual_m3"] == 0.0
    assert math.isclose(
        float(ledger["liquid_to_horizontal_m3"]),
        -float(ledger["reservoir_liquid_change_m3"]),
        rel_tol=0.0,
        abs_tol=1.0e-18,
    )


def test_characteristic_uses_physical_dt_not_valve_hydraulic_time() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = set_reservoir_adjacent_uniform_head(
        solver,
        solver.initial_state(),
        0.40,
    )
    state = replace(state, time=0.01)

    first = boundary.solve(state, dt=1.0e-4)
    # Changing only global experiment time changes the adapter's valve-area
    # approximation, but must not change this independent reservoir boundary.
    later = boundary.solve(replace(state, time=10.0), dt=1.0e-4)

    assert later == first


def test_rejects_a_step_beyond_one_acoustic_cell() -> None:
    solver = build_horizontal()
    boundary = Case1ConstantHeadBoundary(solver)
    state = solver.initial_state()

    with pytest.raises(ValueError, match="more than one grid interval"):
        boundary.solve(state, dt=1.01 * solver.dx / solver.section.wave_speed)
