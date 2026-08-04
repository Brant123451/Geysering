"""Conservation and reference-pressure tests for the compressible T node."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    PRODUCTION_READY,
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    CompressibleNodeBranchRates,
    InconsistentFluxPressureError,
    euler_compressible_finite_node_stage,
    liquid_storage_factor,
    solve_compressible_node_pressure,
    state_from_pressure_and_gas_mass,
)


P_ATM = 101_325.0
RHO_L = 998.0
A_NODE = 28.0
C_G = math.sqrt(287.05 * 293.0)
V_NODE = 1.0e-3


def _params() -> CompressibleFiniteNodeParameters:
    return CompressibleFiniteNodeParameters(
        gas_sound_speed=C_G,
        liquid_density=RHO_L,
        liquid_wave_speed=A_NODE,
        reference_pressure_abs=P_ATM,
    )


def _zero_rates(pressure: float) -> CompressibleNodeBranchRates:
    return CompressibleNodeBranchRates(0.0, 0.0, pressure)


def _occupied_state(
    pressure: float = 107_000.0,
    gas_volume_fraction: float = 0.20,
) -> CompressibleFiniteNodeState:
    params = _params()
    gas_mass = (
        pressure * gas_volume_fraction * V_NODE / params.gas_sound_speed**2
    )
    return state_from_pressure_and_gas_mass(
        pressure_abs=pressure,
        gas_mass=gas_mass,
        node_total_volume=V_NODE,
        params=params,
    )


def test_storage_reference_matches_casea_head_area_law() -> None:
    """p_ref must be atmospheric absolute pressure, not zero gauge pressure."""

    params = _params()
    gravity = 9.81
    diameter = 0.094
    crown_head = diameter + 0.305
    pressure = P_ATM + RHO_L * gravity * (crown_head - diameter)
    expected = 1.0 + gravity * (crown_head - diameter) / A_NODE**2
    assert liquid_storage_factor(pressure, params) == pytest.approx(
        expected, rel=2.0e-15, abs=0.0
    )

    # With no gas, that elastic inventory alone occupies the complete node
    # and recovers the same finite absolute pressure.
    state = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=V_NODE * expected,
        node_total_volume=V_NODE,
    )
    solved = solve_compressible_node_pressure(state, params)
    assert solved.pressure_abs == pytest.approx(pressure, rel=2.0e-14)
    assert solved.gas_physical_volume == 0.0
    assert solved.liquid_physical_volume == pytest.approx(V_NODE)


def test_stationary_three_branch_stage_is_exactly_stationary() -> None:
    params = _params()
    state = _occupied_state()
    pressure = solve_compressible_node_pressure(state, params)
    zero = _zero_rates(pressure.pressure_abs)
    result = euler_compressible_finite_node_stage(
        state,
        1.0e-3,
        west=zero,
        east=zero,
        vertical=zero,
        params=params,
    )
    assert PRODUCTION_READY
    assert result.state == state
    assert result.pressure.pressure_abs == pressure.pressure_abs
    assert result.ledger.gas_mass_balance_residual == 0.0
    assert result.ledger.liquid_inventory_balance_residual == 0.0
    assert result.ledger.fixed_geometric_volume_change == 0.0


def test_exact_zero_gas_launch_uses_elastic_liquid_pressure_then_accepts_inflow() -> None:
    params = _params()
    launch_pressure = 105_500.0
    factor = liquid_storage_factor(launch_pressure, params)
    state = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=V_NODE * factor,
        node_total_volume=V_NODE,
    )
    initial = solve_compressible_node_pressure(state, params)
    assert initial.pressure_abs == pytest.approx(launch_pressure, rel=2.0e-14)

    # Negative outward gas rate is physical inflow from the west branch.  No
    # artificial seed gas, fill volume, or pressure assignment is required.
    west = CompressibleNodeBranchRates(
        gas_mass_outward=-2.0e-4,
        liquid_equivalent_volume_outward=1.0e-5,
        evaluation_pressure_abs=initial.pressure_abs,
    )
    east = CompressibleNodeBranchRates(
        gas_mass_outward=0.0,
        liquid_equivalent_volume_outward=-4.0e-6,
        evaluation_pressure_abs=initial.pressure_abs,
    )
    vertical = CompressibleNodeBranchRates(
        gas_mass_outward=0.0,
        liquid_equivalent_volume_outward=-1.0e-6,
        evaluation_pressure_abs=initial.pressure_abs,
    )
    dt = 1.0e-3
    result = euler_compressible_finite_node_stage(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    assert result.state.gas_mass == pytest.approx(2.0e-7)
    assert result.pressure.gas_physical_volume > 0.0
    assert result.pressure.liquid_physical_volume > 0.0
    assert result.ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=2.0e-23
    )
    assert abs(result.ledger.final_occupancy_residual) <= 2.0e-15


def test_elastic_liquid_inventory_is_conserved_without_cell_reassignment() -> None:
    params = _params()
    state = _occupied_state(pressure=108_000.0, gas_volume_fraction=0.12)
    pressure = solve_compressible_node_pressure(state, params).pressure_abs
    west = CompressibleNodeBranchRates(0.0, 2.0e-5, pressure)
    east = CompressibleNodeBranchRates(0.0, -3.0e-6, pressure)
    vertical = CompressibleNodeBranchRates(0.0, 8.0e-6, pressure)
    dt = 2.5e-3
    result = euler_compressible_finite_node_stage(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    total_outward = 2.0e-5 - 3.0e-6 + 8.0e-6
    expected = state.liquid_equivalent_volume - dt * total_outward
    assert result.state.liquid_equivalent_volume == pytest.approx(
        expected, rel=0.0, abs=2.0e-19
    )
    assert result.ledger.actual_liquid_equivalent_volume_change == pytest.approx(
        -dt * total_outward, rel=0.0, abs=2.0e-19
    )
    assert result.ledger.liquid_inventory_balance_residual == pytest.approx(
        0.0, abs=2.0e-19
    )


def test_mixed_phase_pressure_satisfies_gas_eos_and_fixed_occupancy() -> None:
    params = _params()
    target_pressure = 109_250.0
    state = _occupied_state(target_pressure, gas_volume_fraction=0.37)
    solved = solve_compressible_node_pressure(state, params)
    assert solved.pressure_abs == pytest.approx(target_pressure, rel=2.0e-12)
    assert solved.gas_physical_volume == pytest.approx(
        state.gas_mass * params.gas_sound_speed**2 / solved.pressure_abs,
        rel=2.0e-15,
    )
    assert solved.liquid_physical_volume == pytest.approx(
        state.liquid_equivalent_volume / solved.liquid_storage_factor,
        rel=2.0e-15,
    )
    assert (
        solved.gas_physical_volume + solved.liquid_physical_volume
    ) == pytest.approx(state.node_total_volume, abs=2.0e-15)


def test_three_branch_gas_liquid_and_occupancy_ledgers_close() -> None:
    params = _params()
    state = _occupied_state(pressure=106_700.0, gas_volume_fraction=0.25)
    initial = solve_compressible_node_pressure(state, params)
    pressure = initial.pressure_abs
    west = CompressibleNodeBranchRates(-3.0e-5, 1.0e-5, pressure)
    east = CompressibleNodeBranchRates(8.0e-6, -2.0e-6, pressure)
    vertical = CompressibleNodeBranchRates(4.0e-6, 5.0e-6, pressure)
    dt = 4.0e-3
    result = euler_compressible_finite_node_stage(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )

    gas_out = -3.0e-5 + 8.0e-6 + 4.0e-6
    liquid_out = 1.0e-5 - 2.0e-6 + 5.0e-6
    assert result.ledger.gas_mass_outward_rate == pytest.approx(gas_out)
    assert result.ledger.liquid_equivalent_volume_outward_rate == pytest.approx(
        liquid_out
    )
    assert result.state.gas_mass == pytest.approx(
        state.gas_mass - dt * gas_out
    )
    assert result.state.liquid_equivalent_volume == pytest.approx(
        state.liquid_equivalent_volume - dt * liquid_out
    )
    assert result.ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=2.0e-20
    )
    assert result.ledger.liquid_inventory_balance_residual == pytest.approx(
        0.0, abs=2.0e-19
    )
    assert result.ledger.fixed_geometric_volume_change == 0.0
    assert abs(result.ledger.initial_occupancy_residual) <= 2.0e-15
    assert abs(result.ledger.final_occupancy_residual) <= 2.0e-15


def test_stage_rejects_branch_rates_from_different_node_pressures() -> None:
    params = _params()
    state = _occupied_state()
    pressure = solve_compressible_node_pressure(state, params).pressure_abs
    zero = _zero_rates(pressure)
    wrong = _zero_rates(pressure + 1.0)
    with pytest.raises(InconsistentFluxPressureError):
        euler_compressible_finite_node_stage(
            state,
            1.0e-3,
            west=zero,
            east=wrong,
            vertical=zero,
            params=params,
        )
