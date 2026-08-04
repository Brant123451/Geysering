"""Local conservation tests for the Case-A first-arrival launch closure."""

from __future__ import annotations

import math
from pathlib import Path
import sys


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_gas_coupled_front import GasCellTrace  # noqa: E402
from casea_material_front_cutcell import (  # noqa: E402
    ALEInterfaceFlux,
    PressurisedState,
)
from casea_twofront_launch_closure import (  # noqa: E402
    evaluate_twofront_launch_candidate,
    solve_twofront_launch_closure,
)
from casea_tjunction_shock_network import (  # noqa: E402
    BranchGeometry,
    LiquidCharacteristic,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    solve_zero_storage_t_node,
)


RHO_L = 998.0
G = 9.81
P_ATM = 101_325.0
C_G = math.sqrt(287.05 * 293.0)


class _Fixture:
    def __init__(self) -> None:
        self.east = BranchGeometry(0.094, 0.490, 28.0, 0.0)
        # The algebraic launch closure is orientation-independent.  A zero
        # slope gives an exact stationary regression; production may pass the
        # vertical bed source and a nonzero local dt through the same API.
        self.vertical = BranchGeometry(0.0571, 0.610, 28.0, 0.0)
        self.gas_head = 0.20
        self.pressure = (
            P_ATM + RHO_L * G * self.gas_head
        )
        self.east_depth = 0.070
        self.vertical_depth = 0.040
        self.east_area = float(
            self.east.section(G).area_from_depth(self.east_depth)
        )
        self.vertical_area = float(
            self.vertical.section(G).area_from_depth(self.vertical_depth)
        )
        self.areas = ZeroStorageTBranchAreas(
            west=self.east.section(G).full_area,
            east=self.east.section(G).full_area,
            vertical=self.vertical.section(G).full_area,
        )
        characteristic = LiquidCharacteristic(self.pressure, 0.0, 28.0)
        self.characteristics = TeeLiquidCharacteristics(
            west=characteristic,
            east=characteristic,
            vertical=characteristic,
            west_liquid_area=self.areas.west,
        )
        self.west_gas_area = (
            self.east.section(G).full_area - self.east_area
        )

    def foot(
        self,
        geometry: BranchGeometry,
        depth: float,
        *,
        gas_head: float | None = None,
    ) -> PressurisedState:
        head = self.gas_head if gas_head is None else gas_head
        section = geometry.section(G)
        pressurised_head = (
            head
            + 0.5 * geometry.diameter
            + float(section.hydrostatic_moment(depth)) / section.full_area
        )
        return PressurisedState(
            area=float(section.area_from_head(pressurised_head)),
            discharge=0.0,
        )

    def solve(
        self,
        gas_velocity: float,
        *,
        vertical_foot: PressurisedState | None = None,
    ):
        return solve_twofront_launch_closure(
            west_gas_trace=GasCellTrace(
                density=self.pressure / C_G**2,
                velocity=gas_velocity,
                sound_speed=C_G,
            ),
            west_gas_face_area=self.west_gas_area,
            liquid_characteristics=self.characteristics,
            liquid_areas=self.areas,
            east_geometry=self.east,
            vertical_geometry=self.vertical,
            east_pressurised_foot=self.foot(
                self.east, self.east_depth
            ),
            vertical_pressurised_foot=(
                self.foot(self.vertical, self.vertical_depth)
                if vertical_foot is None
                else vertical_foot
            ),
            east_stratified_liquid_area=self.east_area,
            vertical_stratified_liquid_area=self.vertical_area,
        )


def test_hydrostatic_rest_launch_is_stationary_and_well_balanced() -> None:
    fixture = _Fixture()
    result = fixture.solve(0.0)
    assert abs(result.gas_node_pressure_abs - fixture.pressure) < 1.0e-7
    assert abs(result.east_traces.speed) < 1.0e-8
    assert abs(result.vertical_traces.speed) < 1.0e-8
    assert abs(result.west_gas_characteristic_mass_inflow) < 1.0e-12
    assert abs(result.gas_mass_balance_residual) < 1.0e-12
    assert abs(result.east_liquid_rh_residual) < 1.0e-8
    assert abs(result.vertical_liquid_rh_residual) < 1.0e-8


def test_west_characteristic_launches_both_fronts_without_a_split_rule() -> None:
    fixture = _Fixture()
    result = fixture.solve(0.20)
    assert result.east_traces.speed > 0.0
    assert result.vertical_traces.speed > 0.0
    assert result.east_gas_volume_creation_rate > 0.0
    assert result.vertical_gas_volume_creation_rate > 0.0
    # The unequal areas/RH states determine an unequal physical split.
    assert not math.isclose(
        result.east_gas_mass_demand_rate,
        result.vertical_gas_mass_demand_rate,
        rel_tol=1.0e-3,
    )


def test_one_branch_can_remain_stationary_while_the_other_launches() -> None:
    fixture = _Fixture()
    target_pressure = fixture.pressure + 100.0
    target_head = (
        fixture.gas_head + 100.0 / (RHO_L * G)
    )
    vertical_foot = fixture.foot(
        fixture.vertical,
        fixture.vertical_depth,
        gas_head=target_head,
    )
    liquid_node = solve_zero_storage_t_node(
        fixture.characteristics,
        fixture.areas,
        liquid_density=RHO_L,
    )
    zero_velocity_trace = GasCellTrace(
        fixture.pressure / C_G**2, 0.0, C_G
    )
    target = evaluate_twofront_launch_candidate(
        gas_node_pressure_abs=target_pressure,
        west_gas_trace=zero_velocity_trace,
        west_gas_face_area=fixture.west_gas_area,
        liquid_node=liquid_node,
        east_geometry=fixture.east,
        vertical_geometry=fixture.vertical,
        east_pressurised_foot=fixture.foot(
            fixture.east, fixture.east_depth
        ),
        vertical_pressurised_foot=vertical_foot,
        east_stratified_liquid_area=fixture.east_area,
        vertical_stratified_liquid_area=fixture.vertical_area,
    )
    density_target = target_pressure / C_G**2
    required_face_velocity = (
        target.gas_mass_demand_rate
        / (density_target * fixture.west_gas_area)
    )
    trace_velocity = (
        required_face_velocity
        + (target_pressure - fixture.pressure)
        / zero_velocity_trace.acoustic_impedance
    )
    result = fixture.solve(trace_velocity, vertical_foot=vertical_foot)
    assert abs(result.gas_node_pressure_abs - target_pressure) < 1.0e-6
    assert result.east_traces.speed > 0.0
    assert abs(result.vertical_traces.speed) < 1.0e-8
    assert result.east_gas_mass_demand_rate > 0.0
    assert abs(result.vertical_gas_mass_demand_rate) < 1.0e-12


def test_gas_liquid_and_ale_ledgers_close_at_launch() -> None:
    fixture = _Fixture()
    result = fixture.solve(0.20)
    assert math.isclose(
        result.west_gas_characteristic_mass_inflow,
        result.east_gas_mass_demand_rate
        + result.vertical_gas_mass_demand_rate,
        rel_tol=1.0e-9,
        abs_tol=2.0e-12,
    )
    assert abs(result.gas_mass_balance_residual) < 2.0e-12
    assert abs(result.liquid_volume_balance_residual) < 1.0e-12
    assert abs(result.liquid_mass_balance_residual) < 1.0e-9
    for traces in (result.east_traces, result.vertical_traces):
        ale = ALEInterfaceFlux.from_traces(traces)
        assert abs(ale.liquid_area_residual) < 1.0e-11
        assert abs(ale.liquid_momentum_residual) < 1.0e-10
        assert ale.gas_mass == 0.0
        assert ale.gas_material_residual == 0.0


def test_more_west_gas_momentum_raises_shared_pressure_and_front_speeds() -> None:
    fixture = _Fixture()
    slow = fixture.solve(0.10)
    fast = fixture.solve(0.30)
    assert fast.gas_node_pressure_abs > slow.gas_node_pressure_abs
    assert fast.east_traces.speed > slow.east_traces.speed
    assert fast.vertical_traces.speed > slow.vertical_traces.speed

