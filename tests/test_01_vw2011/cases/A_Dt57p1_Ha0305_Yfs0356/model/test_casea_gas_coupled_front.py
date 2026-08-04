"""Regression tests for the gas-characteristic material-front closure."""

from __future__ import annotations

import math
from pathlib import Path
import sys


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_gas_coupled_front import (  # noqa: E402
    GasCellTrace,
    solve_gas_coupled_material_front,
)
from casea_tjunction_shock_network import (  # noqa: E402
    BranchGeometry,
    MovingFrontState,
    solve_front_rankine_hugoniot,
)


RHO_L = 998.0
G = 9.81
P_ATM = 101_325.0
C_G = math.sqrt(287.05 * 293.0)


def _fixture() -> tuple[BranchGeometry, MovingFrontState, float]:
    geometry = BranchGeometry(
        diameter=0.094,
        length=0.490,
        wave_speed=28.0,
    )
    equilibrium_head = 0.3780169412545817
    front = MovingFrontState(
        position=0.10,
        free_surface_depth=0.070,
        pressurised_head_foot=0.450,
    )
    equilibrium_pressure = P_ATM + RHO_L * G * equilibrium_head
    return geometry, front, equilibrium_pressure


def _trace(pressure: float, velocity: float = 0.0) -> GasCellTrace:
    return GasCellTrace(
        density=pressure / C_G**2,
        velocity=velocity,
        sound_speed=C_G,
    )


def test_stationary_rh_state_is_exact_gas_characteristic_equilibrium() -> None:
    geometry, front, pressure = _fixture()
    result = solve_gas_coupled_material_front(
        front,
        geometry,
        gas_trace=_trace(pressure),
        atmospheric_pressure=P_ATM,
        liquid_density=RHO_L,
        gravity=G,
        free_surface_velocity=0.0,
    )
    assert abs(result.interface_speed) < 2.0e-10
    assert abs(result.gas_pressure_abs - pressure) < 2.0e-7
    assert abs(result.characteristic_residual) < 2.0e-7
    assert result.relative_gas_mass_flux_per_area == 0.0


def test_matching_cell_velocity_recovers_fixed_pressure_rh_solution() -> None:
    geometry, front, equilibrium_pressure = _fixture()
    cell_pressure = equilibrium_pressure + RHO_L * G * 0.05
    fixed = solve_front_rankine_hugoniot(
        front,
        geometry,
        gas_pressure_abs=cell_pressure,
        atmospheric_pressure=P_ATM,
        liquid_density=RHO_L,
        gravity=G,
        free_surface_velocity=0.0,
    )
    result = solve_gas_coupled_material_front(
        front,
        geometry,
        gas_trace=_trace(cell_pressure, fixed.interface_speed),
        atmospheric_pressure=P_ATM,
        liquid_density=RHO_L,
        gravity=G,
        free_surface_velocity=0.0,
    )
    assert abs(result.gas_pressure_abs - cell_pressure) < 2.0e-7
    assert abs(result.interface_speed - fixed.interface_speed) < 2.0e-10


def test_advancing_and_receding_fronts_satisfy_gas_impedance_relation() -> None:
    geometry, front, equilibrium_pressure = _fixture()
    results = []
    cell_pressures = []
    for head_offset in (-0.05, 0.05):
        cell_pressure = equilibrium_pressure + RHO_L * G * head_offset
        trace = _trace(cell_pressure)
        result = solve_gas_coupled_material_front(
            front,
            geometry,
            gas_trace=trace,
            atmospheric_pressure=P_ATM,
            liquid_density=RHO_L,
            gravity=G,
            free_surface_velocity=0.0,
        )
        reconstructed = (
            trace.pressure_abs
            + trace.acoustic_impedance
            * (result.interface_speed - trace.velocity)
        )
        assert abs(result.gas_pressure_abs - reconstructed) < 2.0e-5
        assert result.liquid.residual_linf < 1.0e-7
        results.append(result)
        cell_pressures.append(cell_pressure)
    assert results[0].interface_speed < -0.08
    assert results[0].gas_pressure_abs < cell_pressures[0]
    assert results[1].interface_speed > 0.08
    assert results[1].gas_pressure_abs > cell_pressures[1]


def test_gas_pressure_correction_has_physical_compression_sign() -> None:
    geometry, front, equilibrium_pressure = _fixture()
    cell_pressure = equilibrium_pressure + RHO_L * G * 0.05
    trace = _trace(cell_pressure, velocity=0.0)
    result = solve_gas_coupled_material_front(
        front,
        geometry,
        gas_trace=trace,
        atmospheric_pressure=P_ATM,
        liquid_density=RHO_L,
        gravity=G,
        free_surface_velocity=0.0,
    )
    assert result.interface_speed > trace.velocity
    assert result.gas_pressure_abs > trace.pressure_abs
