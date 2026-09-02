from __future__ import annotations

import math

import pytest

from casea_compressible_finite_node import (
    CompressibleFiniteNodeParameters,
    CompressibleFiniteNodeState,
)
from casea_compressible_node_launch import (
    CompressibleNodeLaunchError,
    LaunchStepCrossesNodeBranchPoint,
    advance_compressible_node_west_launch,
)
from casea_material_front_cutcell import PressurisedState, StratifiedState
from casea_paper_material_front_rh import PaperFrontPhysics
from casea_tjunction_shock_network import (
    LiquidCharacteristic,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
)


RHO_L = 998.0
P_ATM = 101_325.0
A_LIQUID = 28.0
C_GAS = math.sqrt(287.05 * 293.0)
D_H = 0.094
D_V = 0.0571
A_H = math.pi * D_H**2 / 4.0
A_V = math.pi * D_V**2 / 4.0


def _case_a_event() -> tuple[CompressibleFiniteNodeState, dict[str, object]]:
    west_liquid_area = 0.003121443135478332
    west_gas_mass = 0.004834739061124715
    west_gas_momentum = 0.0022156928802623535
    west_liquid_discharge = -0.0017922203114238986
    east = PressurisedState(
        area=0.006992983436657241,
        discharge=-0.00013253326141847847,
    )

    def elastic_pressure(area: float, full: float) -> float:
        return P_ATM + RHO_L * A_LIQUID**2 * (area / full - 1.0)

    west_gas_area = A_H - west_liquid_area
    west_pressure = west_gas_mass / west_gas_area * C_GAS**2
    characteristics = TeeLiquidCharacteristics(
        west=LiquidCharacteristic(
            reference_pressure_abs=west_pressure,
            reference_outward_velocity=(
                -west_liquid_discharge / west_liquid_area
            ),
            wave_speed=A_LIQUID,
        ),
        east=LiquidCharacteristic(
            reference_pressure_abs=elastic_pressure(east.area, A_H),
            reference_outward_velocity=east.discharge / east.area,
            wave_speed=A_LIQUID,
        ),
        vertical=LiquidCharacteristic(
            reference_pressure_abs=P_ATM + RHO_L * 9.81 * 0.356,
            reference_outward_velocity=0.0,
            wave_speed=A_LIQUID,
            loss_coefficient=0.75,
        ),
        west_liquid_area=west_liquid_area,
    )
    state = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=0.00041489943273579884,
        node_total_volume=0.0004117370389316963,
    )
    kwargs: dict[str, object] = {
        "node_params": CompressibleFiniteNodeParameters(
            gas_sound_speed=C_GAS,
            liquid_density=RHO_L,
            liquid_wave_speed=A_LIQUID,
        ),
        "west_pressurised_foot": east,
        "west_stratified_foot": StratifiedState(
            gas_mass=west_gas_mass,
            gas_momentum=west_gas_momentum,
            liquid_area=west_liquid_area,
            liquid_discharge=west_liquid_discharge,
        ),
        "west_physics": PaperFrontPhysics(
            diameter=D_H,
            liquid_wave_speed=A_LIQUID,
            liquid_density=RHO_L,
            gravity=9.81,
            reference_pressure=P_ATM,
            gas_sound_speed=C_GAS,
            cos_inclination=1.0,
        ),
        "liquid_characteristics": characteristics,
        "liquid_areas": ZeroStorageTBranchAreas(
            west=west_liquid_area,
            east=A_H,
            vertical=A_V,
        ),
        # event face 3.48522 m to measured tower centre 3.516 m
        "distance_to_first_branch": 3.516 - 3.48522,
    }
    return state, kwargs


def test_case_a_launch_enters_only_the_finite_node_from_the_west() -> None:
    state, kwargs = _case_a_event()
    dt = 1.0e-5
    result = advance_compressible_node_west_launch(
        state,
        dt=dt,
        **kwargs,
    )
    assert result.front_candidate.active_set == "middle"
    assert result.front_candidate.speed == pytest.approx(
        0.233369463167545, rel=2.0e-12
    )
    assert result.front_distance > 0.0
    assert result.front_distance < kwargs["distance_to_first_branch"]
    assert result.state.gas_mass == pytest.approx(
        dt * result.west_gas_mass_rate_into_node, abs=2.0e-18
    )
    # There are deliberately no east/vertical gas receiver fields: the gas
    # has not yet crossed either geometric branch opening.
    assert not hasattr(result, "east_receiver_mass_rate")
    assert not hasattr(result, "vertical_receiver_mass_rate")
    assert result.gas_mass_balance_residual == pytest.approx(0.0, abs=2.0e-18)
    assert result.liquid_inventory_balance_residual == pytest.approx(
        0.0, abs=5.0e-18
    )
    assert result.next_pressure.occupancy_residual == pytest.approx(
        0.0, abs=1.0e-14
    )
    # An explicit launch is first-order in time; the EOS occupancy and swept
    # front volume therefore agree to O(dt^2), without any volume assignment.
    assert abs(result.geometric_volume_residual) < 2.0e-10


def test_launch_rejects_a_step_that_skips_the_branch_topology_event() -> None:
    state, kwargs = _case_a_event()
    with pytest.raises(LaunchStepCrossesNodeBranchPoint):
        advance_compressible_node_west_launch(state, dt=1.0, **kwargs)


def test_no_positive_middle_root_does_not_create_receiver_gas() -> None:
    state, kwargs = _case_a_event()
    # A sufficiently rarefied liquid inventory lowers the liquid-full node
    # pressure enough that the continuous middle-family front recedes.  Any
    # other algebraic root must not be selected as a launch.
    params = kwargs["node_params"]
    assert isinstance(params, CompressibleFiniteNodeParameters)
    storage = 1.0 + (95_000.0 - P_ATM) / (
        RHO_L * A_LIQUID**2
    )
    blocked = CompressibleFiniteNodeState(
        gas_mass=0.0,
        liquid_equivalent_volume=(
            state.node_total_volume * storage
        ),
        node_total_volume=state.node_total_volume,
    )
    with pytest.raises(CompressibleNodeLaunchError, match="positive middle"):
        advance_compressible_node_west_launch(blocked, dt=1.0e-5, **kwargs)
