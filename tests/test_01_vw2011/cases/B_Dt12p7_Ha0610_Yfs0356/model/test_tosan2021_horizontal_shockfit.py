"""Focused regression tests for the independent Tosan-2021 Case-B core."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from tosan2021_horizontal_shockfit import (  # noqa: E402
    CircularSection,
    HorizontalConfig,
    PolytropicGasInventory,
    Tosan2021HorizontalShockFit,
    TosanInterfaceData,
    WetDryState,
    advance_shock_position,
    central_upwind_wet_dry_step,
    circular_dry_bed_gate_state,
    solve_tosan_positive_interface,
)


class CircularGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = CircularSection(0.094, wave_speed=100.0)

    def test_area_depth_inverse_and_force_endpoints(self) -> None:
        depths = np.linspace(0.0, self.section.diameter, 17)
        areas = self.section.area_from_depth(depths)
        recovered = self.section.depth_from_area(areas)
        np.testing.assert_allclose(recovered, depths, atol=2.0e-14, rtol=0.0)
        self.assertEqual(self.section.hydrostatic_moment(0.0), 0.0)
        self.assertAlmostEqual(
            self.section.hydrostatic_moment(self.section.diameter),
            np.pi * self.section.diameter**3 / 8.0,
            places=15,
        )

    def test_elastic_branch_is_continuous_and_uses_water_head(self) -> None:
        area_full = self.section.full_area
        self.assertAlmostEqual(
            self.section.pressure_flux(area_full),
            self.section.gravity * self.section.full_hydrostatic_moment,
            places=15,
        )
        case_b_area = self.section.area_from_head(0.356)
        self.assertGreater(case_b_area, area_full)
        self.assertAlmostEqual(
            self.section.head_from_area(case_b_area), 0.356, places=12
        )

    def test_circular_dry_gate_is_finite_and_left_moving(self) -> None:
        depth, velocity = circular_dry_bed_gate_state(
            self.section, direction=-1
        )
        self.assertGreater(depth, 0.0)
        self.assertLess(depth, self.section.diameter)
        self.assertLess(velocity, 0.0)
        self.assertLess(abs(velocity), 2.0 * np.sqrt(9.81 * 0.094))


class GasAndInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = CircularSection(0.094, wave_speed=100.0)

    def test_polytropic_invariant_is_preserved(self) -> None:
        gas = PolytropicGasInventory.from_gauge_head(
            volume=3.789e-3,
            gauge_head=0.610,
            atmospheric_pressure=101_325.0,
            liquid_density=998.2,
            gravity=9.81,
            gamma=1.4,
        )
        expanded = gas.with_volume(1.25 * gas.volume)
        self.assertAlmostEqual(
            expanded.pressure_abs * expanded.volume**expanded.gamma,
            gas.invariant,
            delta=1.0e-12 * gas.invariant,
        )
        self.assertEqual(expanded.mass, gas.mass)

    def test_positive_interface_recovers_constructed_645_to_647_state(self) -> None:
        h_fs = 0.040
        u_fs = 0.120
        u_p = 0.180
        gas_head = 0.080
        area = self.section.full_area
        area_fs = self.section.area_from_depth(h_fs)
        moment_fs = self.section.hydrostatic_moment(h_fs)
        speed = (area * u_p - area_fs * u_fs) / (area - area_fs)
        head_p = (
            0.5 * self.section.diameter
            + gas_head
            + (
                self.section.gravity * moment_fs
                - area * (speed - u_fs) * (u_fs - u_p)
            )
            / (self.section.gravity * area)
        )
        velocity_foot = 0.0
        head_foot = head_p + self.section.wave_speed / self.section.gravity * (
            u_p - velocity_foot
        )
        solution = solve_tosan_positive_interface(
            TosanInterfaceData(
                pressurised_velocity_foot=velocity_foot,
                pressurised_head_foot=head_foot,
                free_surface_velocity=u_fs,
                free_surface_depth=h_fs,
                gas_pressure_head=gas_head,
            ),
            section=self.section,
        )
        self.assertTrue(solution.converged)
        self.assertLess(solution.residual_linf, 1.0e-11)
        self.assertAlmostEqual(solution.pressurised_velocity, u_p, places=11)
        self.assertAlmostEqual(solution.pressurised_head, head_p, places=11)
        self.assertAlmostEqual(solution.interface_speed, speed, places=11)

    def test_shock_position_is_explicit_and_clipped(self) -> None:
        self.assertAlmostEqual(
            advance_shock_position(0.5, 0.3, 0.2, length=1.0), 0.56
        )
        self.assertEqual(
            advance_shock_position(0.9, 2.0, 1.0, length=1.0), 1.0
        )


class WetDryAndCaseBCoreTests(unittest.TestCase):
    def test_zero_depth_is_preserved_ahead_of_finite_wetting_front(self) -> None:
        section = CircularSection(0.094, wave_speed=100.0)
        area = np.zeros(80)
        area[40:] = section.area_from_depth(0.060)
        initial_volume = float(np.sum(area))
        state = WetDryState(area, np.zeros_like(area))
        for _ in range(20):
            state = central_upwind_wet_dry_step(
                state,
                dx=0.010,
                dt=0.001,
                section=section,
                dry_area_fraction=1.0e-10,
                left_boundary="wall",
                right_boundary="wall",
            )
        self.assertTrue(np.all(state.area >= 0.0))
        self.assertTrue(np.all(state.area[:15] == 0.0))
        self.assertAlmostEqual(
            float(np.sum(state.area)),
            initial_volume,
            delta=2.0e-10 * initial_volume,
        )

    def test_case_b_split_core_has_no_acoustic_dry_bed_leak(self) -> None:
        config = HorizontalConfig(
            dx=0.020,
            wave_speed=100.0,
            initial_air_head=0.610,
            initial_water_head=0.356,
            dry_area_fraction=1.0e-8,
        )
        model = Tosan2021HorizontalShockFit(config)
        state = model.case_b_initial_state()
        upstream = model.x < config.valve_x
        downstream = ~upstream
        self.assertTrue(np.all(state.area[upstream] == 0.0))
        self.assertTrue(np.all(state.area[downstream] > model.section.full_area))
        self.assertAlmostEqual(
            model.section.head_from_area(state.area[downstream][0]),
            0.356,
            places=12,
        )
        geometric_volume = model._connected_gas_volume(
            state.area,
            state.interface_x,
            model.dry_gate_area,
        )
        # Regression for full-overlap cells being mistaken for the unique cut
        # cell by floating-point roundoff.
        self.assertGreater(geometric_volume, 0.98 * state.gas.volume)

        initial_solution = model._interface_solution(state)
        self.assertTrue(initial_solution.converged)
        self.assertEqual(
            initial_solution.formulation,
            "tosan_negative_flxT1_reflected_case_b",
        )
        self.assertGreater(initial_solution.interface_speed, 0.0)
        self.assertLessEqual(
            initial_solution.interface_speed,
            model.case_b_entropy_speed_bound + 1.0e-12,
        )
        self.assertLess(initial_solution.residual_linf, 1.0e-8)

        advanced = model.step(state, 0.002)
        self.assertTrue(np.all(np.isfinite(advanced.area)))
        self.assertTrue(np.all(np.isfinite(advanced.discharge)))
        self.assertTrue(np.all(advanced.area >= 0.0))
        self.assertEqual(advanced.area[0], 0.0)
        self.assertGreater(advanced.wetting_front_x, 0.40)
        self.assertGreater(advanced.interface_x, config.valve_x)
        self.assertTrue(advanced.nonlinear_converged)
        self.assertLess(advanced.interface_pressurised_head, 2.0)
        self.assertLess(
            (
                advanced.air_pressure_abs
                - config.atmospheric_pressure
            )
            / (config.liquid_density * config.gravity),
            1.0,
        )
        self.assertAlmostEqual(
            advanced.gas.pressure_abs * advanced.gas.volume**config.gamma,
            advanced.gas.invariant,
            delta=1.0e-11 * advanced.gas.invariant,
        )

        snapshot = model.snapshot(advanced)
        required = {
            "area_fraction",
            "interface_x",
            "air_pressure_abs",
            "air_volume",
            "wetting_front_x",
            "numerical_wetting_front_x",
        }
        self.assertTrue(required.issubset(snapshot))
        self.assertEqual(np.asarray(snapshot["area_fraction"]).shape, model.x.shape)
        self.assertLessEqual(
            snapshot["numerical_wetting_front_x"],
            snapshot["wetting_front_x"],
        )
        self.assertAlmostEqual(model.x[-1] + 0.5 * model.dx, config.length)

    def test_pressure_hook_activates_only_after_vent_crossing(self) -> None:
        calls: list[tuple[float, float, float]] = []

        def relax(time: float, interface_x: float, closed_pressure: float) -> float:
            calls.append((time, interface_x, closed_pressure))
            return 101_325.0

        config = HorizontalConfig(
            length=1.0,
            valve_x=0.20,
            vent_x=0.25,
            dx=0.02,
            wave_speed=50.0,
        )
        model = Tosan2021HorizontalShockFit(
            config, vent_pressure_hook=relax
        )
        state = replace(
            model.case_b_initial_state(),
            interface_x=0.30,
            vented=True,
        )
        advanced = model.step(state, 1.0e-4)
        self.assertTrue(advanced.vented)
        self.assertTrue(calls)
        self.assertEqual(advanced.air_pressure_abs, config.atmospheric_pressure)


if __name__ == "__main__":
    unittest.main()
