"""Independent conservation tests for the post-arrival Case-A T graph."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_tjunction_shock_network import (  # noqa: E402
    BranchGeometry,
    IncompatibleZeroStoragePressure,
    LiquidCharacteristic,
    MovingFrontState,
    StepSubdivisionRequired,
    TJunctionParameters,
    TJunctionShockState,
    TeeLiquidCharacteristics,
    ZeroStorageTBranchAreas,
    advance_tjunction_shock_network,
    common_node_pressure,
    evaluate_zero_storage_t_node_at_pressure,
    solve_zero_storage_t_node,
    solve_front_rankine_hugoniot,
)


class CaseATJunctionShockNetworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.east = BranchGeometry(
            diameter=0.094,
            length=0.490,
            wave_speed=28.0,
            bed_slope=0.0,
        )
        # A horizontal orientation is used in the exact stationary-network
        # tests.  The production vertical branch passes bed_slope=1 and is
        # solved by the identical signed RH function.
        self.vertical = BranchGeometry(
            diameter=0.0571,
            length=0.610,
            wave_speed=28.0,
            bed_slope=0.0,
        )
        self.params = TJunctionParameters(
            east=self.east,
            vertical=self.vertical,
            tee_total_volume=3.5e-4,
        )

    def _stationary_front(
        self,
        geometry: BranchGeometry,
        depth: float,
        gas_head: float,
        position: float,
    ) -> MovingFrontState:
        section = geometry.section(self.params.gravity)
        pressure_moment_head = (
            float(section.hydrostatic_moment(depth)) / section.full_area
        )
        pressurised_head = (
            gas_head + 0.5 * geometry.diameter + pressure_moment_head
        )
        return MovingFrontState(
            position=position,
            free_surface_depth=depth,
            pressurised_head_foot=pressurised_head,
            pressurised_velocity_foot=0.0,
        )

    def _equilibrium_fixture(
        self,
        *,
        gas_head: float = 0.20,
        vertical_at_top: bool = False,
    ) -> tuple[
        TJunctionShockState,
        TeeLiquidCharacteristics,
        float,
    ]:
        pressure = (
            self.params.atmospheric_pressure
            + self.params.liquid_density * self.params.gravity * gas_head
        )
        east_front = self._stationary_front(
            self.east, 0.070, gas_head, 0.080
        )
        vertical_front = self._stationary_front(
            self.vertical,
            0.040,
            gas_head,
            self.vertical.length if vertical_at_top else 0.120,
        )
        provisional = TJunctionShockState(
            time=6.5,
            gas_mass=1.0,
            west_gas_volume=2.5e-3,
            tee_gas_volume=1.5e-4,
            east_front=east_front,
            vertical_front=vertical_front,
        )
        volume = (
            provisional.west_gas_volume
            + provisional.tee_gas_volume
            + (
                self.east.section(self.params.gravity).full_area
                - float(
                    self.east.section(self.params.gravity).area_from_depth(
                        east_front.free_surface_depth
                    )
                )
            )
            * east_front.position
            + (
                self.vertical.section(self.params.gravity).full_area
                - float(
                    self.vertical.section(self.params.gravity).area_from_depth(
                        vertical_front.free_surface_depth
                    )
                )
            )
            * vertical_front.position
        )
        state = replace(
            provisional,
            gas_mass=(
                pressure
                * volume
                / (self.params.gas_constant * self.params.gas_temperature)
            ),
        )
        east_liquid_area = float(
            self.east.section(self.params.gravity).area_from_depth(0.070)
        )
        characteristics = TeeLiquidCharacteristics(
            west=LiquidCharacteristic(pressure, 0.0, 28.0),
            east=LiquidCharacteristic(pressure, 0.0, 28.0),
            vertical=LiquidCharacteristic(pressure, 0.0, 28.0),
            west_liquid_area=east_liquid_area,
        )
        return state, characteristics, pressure

    def test_full_rh_front_advances_stagnates_and_recedes(self) -> None:
        gas_head_equilibrium = 0.3780169412545817
        front = MovingFrontState(
            position=0.10,
            free_surface_depth=0.070,
            pressurised_head_foot=0.450,
        )
        pressure_equilibrium = (
            self.params.atmospheric_pressure
            + self.params.liquid_density
            * self.params.gravity
            * gas_head_equilibrium
        )

        def speed(head_offset: float) -> float:
            solution = solve_front_rankine_hugoniot(
                front,
                self.east,
                gas_pressure_abs=(
                    pressure_equilibrium
                    + self.params.liquid_density
                    * self.params.gravity
                    * head_offset
                ),
                atmospheric_pressure=self.params.atmospheric_pressure,
                liquid_density=self.params.liquid_density,
                gravity=self.params.gravity,
                free_surface_velocity=0.0,
            )
            return solution.interface_speed

        self.assertLess(speed(-0.05), -0.08)
        self.assertAlmostEqual(speed(0.0), 0.0, delta=1.0e-10)
        self.assertGreater(speed(+0.05), +0.08)

    def test_zero_storage_node_is_well_balanced_at_hydrostatic_rest(self) -> None:
        pressure = 104_000.0
        characteristic = LiquidCharacteristic(pressure, 0.0, 28.0)
        characteristics = TeeLiquidCharacteristics(
            west=characteristic,
            east=characteristic,
            vertical=characteristic,
            west_liquid_area=1.0,
        )
        areas = ZeroStorageTBranchAreas(
            west=self.east.section(self.params.gravity).full_area,
            east=self.east.section(self.params.gravity).full_area,
            vertical=self.vertical.section(self.params.gravity).full_area,
        )
        result = solve_zero_storage_t_node(
            characteristics,
            areas,
            liquid_density=self.params.liquid_density,
            required_gas_pressure_abs=pressure,
        )
        self.assertAlmostEqual(result.node_pressure_abs, pressure, delta=1.0e-7)
        self.assertAlmostEqual(result.net_outward_volume_flux, 0.0, delta=1.0e-12)
        self.assertAlmostEqual(result.net_outward_mass_flux, 0.0, delta=1.0e-10)
        for name, flux in result.branch_fluxes.items():
            self.assertAlmostEqual(flux.outward_velocity, 0.0, delta=1.0e-12)
            self.assertAlmostEqual(flux.volume_flux, 0.0, delta=1.0e-12)
            self.assertAlmostEqual(
                flux.total_momentum_flux,
                pressure * getattr(areas, name),
                delta=1.0e-10,
            )
            self.assertAlmostEqual(
                flux.momentum_flux_increment, 0.0, delta=1.0e-10
            )

    def test_zero_storage_node_conserves_arbitrary_signed_branch_fluxes(self) -> None:
        characteristics = TeeLiquidCharacteristics(
            # All coordinates point away from the node.  The high west trace
            # therefore gives q_w<0 (west -> node), balanced by east/vertical.
            west=LiquidCharacteristic(106_000.0, 0.0, 28.0, 0.75),
            east=LiquidCharacteristic(101_000.0, 0.0, 28.0, 0.35),
            vertical=LiquidCharacteristic(100_000.0, 0.0, 28.0, 1.10),
            west_liquid_area=1.0,
        )
        areas = ZeroStorageTBranchAreas(
            west=self.east.section(self.params.gravity).full_area,
            east=self.east.section(self.params.gravity).full_area,
            vertical=self.vertical.section(self.params.gravity).full_area,
        )
        result = solve_zero_storage_t_node(
            characteristics,
            areas,
            liquid_density=self.params.liquid_density,
        )
        fluxes = result.branch_fluxes
        self.assertLess(fluxes["west"].volume_flux, 0.0)
        self.assertGreater(fluxes["east"].volume_flux, 0.0)
        self.assertGreater(fluxes["vertical"].volume_flux, 0.0)
        self.assertAlmostEqual(
            math.fsum(flux.volume_flux for flux in fluxes.values()),
            0.0,
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            math.fsum(flux.mass_flux for flux in fluxes.values()),
            0.0,
            delta=1.0e-9,
        )
        for flux in fluxes.values():
            self.assertAlmostEqual(
                flux.mass_flux,
                self.params.liquid_density * flux.volume_flux,
                delta=1.0e-12,
            )
            self.assertAlmostEqual(
                flux.total_momentum_flux,
                flux.advective_momentum_flux + flux.pressure_force,
                delta=1.0e-12,
            )

    def test_incompatible_fixed_gas_pressure_reports_missing_storage_law(self) -> None:
        characteristics = TeeLiquidCharacteristics(
            west=LiquidCharacteristic(106_000.0, 0.0, 28.0),
            east=LiquidCharacteristic(101_000.0, 0.0, 28.0),
            vertical=LiquidCharacteristic(100_000.0, 0.0, 28.0),
            west_liquid_area=1.0,
        )
        areas = ZeroStorageTBranchAreas(0.006, 0.006, 0.0025)
        zero = solve_zero_storage_t_node(
            characteristics,
            areas,
            liquid_density=self.params.liquid_density,
        )
        prescribed = zero.node_pressure_abs + 500.0
        trial = evaluate_zero_storage_t_node_at_pressure(
            characteristics,
            areas,
            node_pressure_abs=prescribed,
            liquid_density=self.params.liquid_density,
        )
        with self.assertRaises(IncompatibleZeroStoragePressure) as caught:
            solve_zero_storage_t_node(
                characteristics,
                areas,
                liquid_density=self.params.liquid_density,
                required_gas_pressure_abs=prescribed,
            )
        self.assertAlmostEqual(
            caught.exception.required_tee_gas_volume_rate,
            trial.net_outward_volume_flux,
            delta=1.0e-15,
        )
        self.assertNotEqual(caught.exception.required_tee_gas_volume_rate, 0.0)

    def test_zero_storage_pressure_drives_signed_east_rh_without_speed_rule(self) -> None:
        gas_head_equilibrium = 0.3780169412545817
        front = MovingFrontState(
            position=0.10,
            free_surface_depth=0.070,
            pressurised_head_foot=0.450,
        )
        areas = ZeroStorageTBranchAreas(0.006, 0.006, 0.0025)
        speeds: list[float] = []
        for head_offset in (-0.05, 0.0, 0.05):
            pressure = (
                self.params.atmospheric_pressure
                + self.params.liquid_density
                * self.params.gravity
                * (gas_head_equilibrium + head_offset)
            )
            characteristic = LiquidCharacteristic(pressure, 0.0, 28.0)
            node = solve_zero_storage_t_node(
                TeeLiquidCharacteristics(
                    west=characteristic,
                    east=characteristic,
                    vertical=characteristic,
                    west_liquid_area=1.0,
                ),
                areas,
                liquid_density=self.params.liquid_density,
                required_gas_pressure_abs=pressure,
            )
            speeds.append(solve_front_rankine_hugoniot(
                front,
                self.east,
                gas_pressure_abs=node.node_pressure_abs,
                atmospheric_pressure=self.params.atmospheric_pressure,
                liquid_density=self.params.liquid_density,
                gravity=self.params.gravity,
                free_surface_velocity=(
                    node.branch_fluxes["east"].outward_velocity
                ),
            ).interface_speed)
        self.assertLess(speeds[0], 0.0)
        self.assertAlmostEqual(speeds[1], 0.0, delta=1.0e-10)
        self.assertGreater(speeds[2], 0.0)

    def test_closed_network_has_one_pressure_and_exact_gas_mass(self) -> None:
        state, characteristics, pressure = self._equilibrium_fixture()
        result = advance_tjunction_shock_network(
            state, self.params, characteristics, dt=1.0e-3
        )
        self.assertAlmostEqual(result.node_pressure_abs, pressure, delta=1.0e-5)
        self.assertAlmostEqual(result.east.solution.interface_speed, 0.0, delta=1.0e-9)
        self.assertAlmostEqual(
            result.vertical.solution.interface_speed, 0.0, delta=1.0e-9
        )
        self.assertAlmostEqual(result.state.gas_mass, state.gas_mass, delta=1.0e-15)
        self.assertAlmostEqual(result.gas_mass_conservation_error, 0.0, delta=1.0e-15)
        self.assertAlmostEqual(result.eos_residual, 0.0, delta=1.0e-8)
        self.assertAlmostEqual(
            math.fsum(result.branch_gas_masses.values()),
            result.state.gas_mass,
            delta=1.0e-12,
        )
        for boundary_pressure in result.branch_boundary_pressures_abs.values():
            self.assertAlmostEqual(
                boundary_pressure, result.node_pressure_abs, delta=0.0
            )

    def test_tee_and_west_volumes_use_the_same_signed_face_flux(self) -> None:
        state, characteristics, pressure = self._equilibrium_fixture()
        driven = replace(
            characteristics,
            west=LiquidCharacteristic(
                reference_pressure_abs=pressure - 800.0,
                reference_outward_velocity=0.0,
                wave_speed=28.0,
                loss_coefficient=0.75,
            ),
        )
        dt = 2.0e-4
        result = advance_tjunction_shock_network(
            state, self.params, driven, dt=dt
        )
        q_w = result.liquid_branch_flows["west"]
        q_sum = math.fsum(result.liquid_branch_flows.values())
        self.assertGreater(q_w, 0.0)
        self.assertAlmostEqual(
            result.state.west_gas_volume,
            state.west_gas_volume - q_w * dt,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            result.state.tee_gas_volume,
            state.tee_gas_volume + q_sum * dt,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(result.gas_mass_conservation_error, 0.0, delta=1.0e-15)

    def test_open_vertical_top_uses_a_riemann_flux_and_conserves_mass(self) -> None:
        state, characteristics, _ = self._equilibrium_fixture(
            gas_head=0.40,
            vertical_at_top=True,
        )
        dt = 1.0e-5
        result = advance_tjunction_shock_network(
            state, self.params, characteristics, dt=dt
        )
        self.assertIsNone(result.vertical.solution)
        self.assertEqual(result.state.vertical_front.position, self.vertical.length)
        self.assertGreater(result.top_mass_transfer, 0.0)
        self.assertAlmostEqual(
            result.state.gas_mass + result.top_mass_transfer,
            state.gas_mass,
            delta=1.0e-14,
        )
        self.assertAlmostEqual(result.gas_mass_conservation_error, 0.0, delta=1.0e-14)

    def test_vertical_opens_at_current_liquid_surface_not_riser_rim(self) -> None:
        state, characteristics, _ = self._equilibrium_fixture(
            gas_head=0.40,
            vertical_at_top=False,
        )
        self.assertLess(state.vertical_front.position, self.vertical.length)
        result = advance_tjunction_shock_network(
            state,
            self.params,
            characteristics,
            dt=1.0e-7,
            vertical_liquid_surface_height=state.vertical_front.position,
        )
        self.assertIsNone(result.vertical.solution)
        self.assertGreater(result.top_mass_transfer, 0.0)
        self.assertAlmostEqual(
            result.state.gas_mass + result.top_mass_transfer,
            state.gas_mass,
            delta=1.0e-14,
        )

    def test_crossing_current_surface_requests_event_subdivision(self) -> None:
        state, characteristics, _ = self._equilibrium_fixture(gas_head=0.20)
        driven = replace(state, gas_mass=1.01 * state.gas_mass)
        with self.assertRaises(StepSubdivisionRequired):
            advance_tjunction_shock_network(
                driven,
                self.params,
                characteristics,
                dt=2.0e-3,
                vertical_liquid_surface_height=(
                    state.vertical_front.position + 1.0e-5
                ),
            )


if __name__ == "__main__":
    unittest.main()
