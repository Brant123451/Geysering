"""Independent acceptance tests for the finite-node SSP--RK2 wrapper."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_compressible_finite_node import (  # noqa: E402
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
    solve_compressible_node_pressure,
    state_from_pressure_and_gas_mass,
)
from casea_compressible_node_postlaunch_stage import (  # noqa: E402
    CompressibleNodeResolvedBranch,
    CompressiblePostLaunchParameters,
)
from casea_compressible_node_ssprk2 import (  # noqa: E402
    PRODUCTION_READY,
    ssprk2_compressible_node_postlaunch_step,
)
from casea_material_front_cutcell import StratifiedState  # noqa: E402
from casea_tjunction_shock_network import LiquidCharacteristic  # noqa: E402


P_ATM = 101_325.0
RHO_L = 998.0
A_NODE = 28.0
R_G = 287.05
T_G = 293.0
C_G = math.sqrt(R_G * T_G)
V_NODE = 0.10
FULL_AREA = 0.010
LIQUID_AREA = 0.005
GAS_AREA = FULL_AREA - LIQUID_AREA


def _params() -> CompressiblePostLaunchParameters:
    node = CompressibleFiniteNodeParameters(
        gas_sound_speed=C_G,
        liquid_density=RHO_L,
        liquid_wave_speed=A_NODE,
        reference_pressure_abs=P_ATM,
    )
    return CompressiblePostLaunchParameters(
        node=node,
        gas_constant=R_G,
        gas_temperature=T_G,
        atmospheric_pressure_abs=P_ATM,
    )


def _node_state(
    pressure: float = P_ATM,
    gas_volume_fraction: float = 0.40,
) -> CompressibleFiniteNodeState:
    params = _params().node
    gas_mass = (
        pressure * gas_volume_fraction * V_NODE / params.gas_sound_speed**2
    )
    return state_from_pressure_and_gas_mass(
        pressure_abs=pressure,
        gas_mass=gas_mass,
        node_total_volume=V_NODE,
        params=params,
    )


def _branch(
    *,
    node_pressure: float = P_ATM,
    gas_density_ratio: float = 1.0,
    gas_outward_velocity: float = 0.0,
    liquid_reference_pressure: float | None = None,
    liquid_reference_velocity: float = 0.0,
) -> CompressibleNodeResolvedBranch:
    density = gas_density_ratio * node_pressure / C_G**2
    gas_mass = density * GAS_AREA
    reference_pressure = (
        node_pressure
        if liquid_reference_pressure is None
        else liquid_reference_pressure
    )
    return CompressibleNodeResolvedBranch(
        resolved=StratifiedState(
            gas_mass=gas_mass,
            gas_momentum=gas_mass * gas_outward_velocity,
            liquid_area=LIQUID_AREA,
            liquid_discharge=LIQUID_AREA * liquid_reference_velocity,
        ),
        liquid_characteristic=LiquidCharacteristic(
            reference_pressure_abs=reference_pressure,
            reference_outward_velocity=liquid_reference_velocity,
            wave_speed=A_NODE,
        ),
        liquid_face_area=LIQUID_AREA,
        full_area=FULL_AREA,
        reference_liquid_face_pressure_abs=reference_pressure,
        reference_liquid_pressure_potential=0.0,
    )


def test_ssprk2_preserves_uniform_hydrostatic_rest() -> None:
    params = _params()
    state = _node_state()
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    branch = _branch(node_pressure=pressure)

    result = ssprk2_compressible_node_postlaunch_step(
        state,
        1.0e-4,
        west=branch,
        east=branch,
        vertical=branch,
        params=params,
    )

    # This wrapper is acceptance scaffolding and is intentionally not yet
    # connected to the production network loop.
    assert PRODUCTION_READY is False
    assert result.state == state
    for flux in result.branch_fluxes.values():
        assert flux.gas_mass == pytest.approx(0.0, abs=2.0e-17)
        assert flux.gas_momentum == pytest.approx(0.0, abs=2.0e-12)
        assert flux.liquid_area == 0.0
        assert flux.liquid_momentum == 0.0
    assert result.ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=2.0e-18
    )
    assert result.ledger.liquid_inventory_balance_residual == 0.0


def test_three_branch_time_average_fluxes_close_both_phase_ledgers() -> None:
    params = _params()
    state = _node_state(pressure=103_000.0, gas_volume_fraction=0.35)
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    west = _branch(
        node_pressure=pressure,
        gas_density_ratio=0.98,
        gas_outward_velocity=0.20,
        liquid_reference_pressure=pressure - 80.0,
        liquid_reference_velocity=0.01,
    )
    east = _branch(
        node_pressure=pressure,
        gas_density_ratio=1.01,
        gas_outward_velocity=-0.10,
        liquid_reference_pressure=pressure + 50.0,
        liquid_reference_velocity=-0.005,
    )
    vertical = _branch(
        node_pressure=pressure,
        gas_density_ratio=1.00,
        gas_outward_velocity=0.05,
        liquid_reference_pressure=pressure,
        liquid_reference_velocity=0.002,
    )
    dt = 5.0e-5

    result = ssprk2_compressible_node_postlaunch_step(
        state,
        dt,
        west=west,
        east=east,
        vertical=vertical,
        params=params,
    )
    gas_outward = math.fsum(
        flux.gas_mass for flux in result.branch_fluxes.values()
    )
    liquid_outward = math.fsum(
        flux.liquid_area for flux in result.branch_fluxes.values()
    )

    assert result.state.gas_mass - state.gas_mass == pytest.approx(
        -dt * gas_outward, abs=4.0e-18
    )
    assert (
        result.state.liquid_equivalent_volume
        - state.liquid_equivalent_volume
    ) == pytest.approx(-dt * liquid_outward, abs=2.0e-17)
    assert result.ledger.gas_mass_balance_residual == pytest.approx(
        0.0, abs=4.0e-18
    )
    assert result.ledger.liquid_inventory_balance_residual == pytest.approx(
        0.0, abs=2.0e-17
    )
    assert (
        result.pressure.gas_physical_volume
        + result.pressure.liquid_physical_volume
    ) == pytest.approx(V_NODE, abs=2.0e-13)
    assert result.ledger.fixed_geometric_volume_change == 0.0


def test_stronger_horizontal_inflow_monotonically_increases_vertical_response() -> None:
    """Finite storage transmits west inflow to the vertical characteristic.

    This checks a dynamic pressure-mediated response, not a prescribed
    90-degree momentum split: both cases start from the same node pressure, so
    stage-one vertical flux is identical.  Stronger west inflow changes the
    predictor inventory more, and the recomputed stage-two node pressure then
    produces a larger upward vertical liquid flux.
    """

    params = _params()
    state = _node_state()
    pressure = solve_compressible_node_pressure(state, params.node).pressure_abs
    rest = _branch(node_pressure=pressure)
    weak_west_inflow = _branch(
        node_pressure=pressure,
        liquid_reference_velocity=-0.01,
    )
    strong_west_inflow = _branch(
        node_pressure=pressure,
        liquid_reference_velocity=-0.10,
    )
    dt = 1.0e-3

    weak = ssprk2_compressible_node_postlaunch_step(
        state,
        dt,
        west=weak_west_inflow,
        east=rest,
        vertical=rest,
        params=params,
    )
    strong = ssprk2_compressible_node_postlaunch_step(
        state,
        dt,
        west=strong_west_inflow,
        east=rest,
        vertical=rest,
        params=params,
    )

    assert weak.first_stage.vertical.liquid_area == pytest.approx(
        strong.first_stage.vertical.liquid_area, abs=2.0e-18
    )
    assert (
        strong.second_stage.vertical.liquid_area
        > weak.second_stage.vertical.liquid_area
    )
    assert strong.vertical.liquid_area > weak.vertical.liquid_area
    assert strong.first_stage.node.state.liquid_equivalent_volume > (
        weak.first_stage.node.state.liquid_equivalent_volume
    )

