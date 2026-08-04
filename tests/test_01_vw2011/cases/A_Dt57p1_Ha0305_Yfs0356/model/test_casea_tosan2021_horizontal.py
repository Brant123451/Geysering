"""Regression checks for the Case-A side-T shock-fitting adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import (  # noqa: E402
    CASE_A_SHOCKFIT_SOURCE,
    CaseASideTShockFit,
    build_case_a_shockfit_solver,
    case_a_config,
)
from vw2011_network_twofluid import (  # noqa: E402
    _map_external_horizontal_state,
)
import tosan2021_horizontal_shockfit as CASE_A_SHOCKFIT_CORE  # noqa: E402


class CaseAShockFitAdapterTests(unittest.TestCase):
    def test_crossing_step_is_stopped_on_west_side_of_t(self) -> None:
        class LinearEventSolver(CaseASideTShockFit):
            def step(
                self,
                state,
                dt,
                *,
                external_pressure_abs=None,
            ):
                del external_pressure_abs
                speed = 0.40
                return replace(
                    state,
                    time=state.time + dt,
                    interface_x=state.interface_x + speed * dt,
                    interface_speed=speed,
                    vented=(
                        state.interface_x + speed * dt
                        >= self.junction_face_x
                    ),
                )

        solver = LinearEventSolver(case_a_config(dx=0.040))
        state = replace(
            solver.case_b_initial_state(),
            interface_x=solver.junction_face_x - 0.010,
        )
        advanced = solver.step_until_junction(
            state,
            0.050,
            location_tolerance=1.0e-12,
        )
        self.assertTrue(advanced.reached)
        self.assertLessEqual(
            advanced.state.interface_x, solver.junction_face_x
        )
        self.assertLessEqual(
            solver.junction_face_x - advanced.state.interface_x,
            1.0e-12,
        )
        self.assertAlmostEqual(
            advanced.elapsed + advanced.remaining, 0.050, places=15
        )
        self.assertGreater(advanced.remaining, 0.0)

    def test_non_crossing_step_consumes_the_complete_request(self) -> None:
        class LinearEventSolver(CaseASideTShockFit):
            def step(
                self,
                state,
                dt,
                *,
                external_pressure_abs=None,
            ):
                del external_pressure_abs
                return replace(
                    state,
                    time=state.time + dt,
                    interface_x=state.interface_x + 0.10 * dt,
                )

        solver = LinearEventSolver(case_a_config(dx=0.040))
        state = replace(
            solver.case_b_initial_state(),
            interface_x=solver.junction_face_x - 0.10,
        )
        advanced = solver.step_until_junction(state, 0.020)
        self.assertFalse(advanced.reached)
        self.assertEqual(advanced.elapsed, 0.020)
        self.assertEqual(advanced.remaining, 0.0)

    def test_case_a_parameters_are_mapped_without_changing_the_core(self) -> None:
        config = case_a_config(dx=0.040)
        self.assertTrue(CASE_A_SHOCKFIT_SOURCE.is_file())
        self.assertAlmostEqual(config.length, 4.006)
        self.assertAlmostEqual(config.diameter, 0.094)
        self.assertAlmostEqual(config.valve_x, 0.546)
        self.assertAlmostEqual(config.vent_x, 3.516)
        self.assertAlmostEqual(config.initial_air_head, 0.305)
        self.assertAlmostEqual(config.initial_water_head, 0.094 + 0.356)
        self.assertAlmostEqual(config.wave_speed, 28.0)

    def test_side_t_is_attached_to_the_nearest_finite_volume_face(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.010)
        self.assertLessEqual(
            abs(solver.junction_face_x - solver.config.vent_x),
            0.5 * solver.dx,
        )
        _, inside_cell_is_closed = solver._effective_pressure(
            time=0.0,
            interface_x=solver.config.vent_x,
            closed_pressure_abs=101500.0,
            external_pressure_abs=None,
        )
        _, crossed_face_is_open = solver._effective_pressure(
            time=0.0,
            interface_x=solver.junction_face_x + 1.0e-12,
            closed_pressure_abs=101500.0,
            external_pressure_abs=None,
        )
        self.assertFalse(inside_cell_is_closed)
        self.assertTrue(crossed_face_is_open)

    def test_dry_left_reach_is_not_instantly_flooded(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.040)
        initial = solver.case_b_initial_state()
        completely_left = (
            solver.x + 0.5 * solver.dx
            <= solver.config.valve_x
        )
        self.assertTrue(np.all(initial.area[completely_left] == 0.0))
        cut = int(np.floor(solver.config.valve_x / solver.dx))
        self.assertGreater(initial.area[cut], 0.0)
        self.assertLess(initial.area[cut], solver.section.full_area)

        advanced = solver.step(initial, 0.10)
        self.assertEqual(advanced.area[0], 0.0)
        self.assertGreater(advanced.wetting_front_x, 0.30)
        self.assertGreater(advanced.interface_x, solver.config.valve_x)
        self.assertTrue(np.all(np.isfinite(advanced.area)))
        self.assertTrue(np.all(advanced.area >= 0.0))

    def test_prevent_gas_inventory_is_polytropic_and_mass_conserving(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.040)
        initial = solver.case_b_initial_state()
        advanced = solver.step(initial, 0.10)
        self.assertFalse(advanced.vented)
        self.assertEqual(advanced.gas.mass, initial.gas.mass)
        self.assertAlmostEqual(
            advanced.gas.pressure_abs
            * advanced.gas.volume**solver.config.gamma,
            initial.gas.invariant,
            delta=1.0e-11 * initial.gas.invariant,
        )

    def test_closed_shockfit_branch_conserves_liquid_volume(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.040)
        initial = solver.case_b_initial_state()
        initial_volume = float(np.sum(initial.area) * solver.dx)
        advanced = solver.step(initial, 0.50)
        self.assertAlmostEqual(
            float(np.sum(advanced.area) * solver.dx),
            initial_volume,
            places=13,
        )
        self.assertTrue(np.isfinite(advanced.cumulative_liquid_volume_residual))

    def test_mass_projection_respects_the_acoustic_domain_of_dependence(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.080, wave_speed=28.0)
        initial = solver.case_b_initial_state()
        elapsed = 0.020
        advanced = solver.step(initial, elapsed)
        acoustic_front = (
            solver.config.valve_x + solver.config.wave_speed * elapsed
        )
        remote = solver.x - 0.5 * solver.dx >= acoustic_front
        np.testing.assert_allclose(
            advanced.area[remote],
            initial.area[remote],
            rtol=0.0,
            atol=1.0e-14 * solver.section.full_area,
        )
        self.assertAlmostEqual(
            float(np.sum(advanced.area) * solver.dx),
            float(np.sum(initial.area) * solver.dx),
            places=13,
        )

    def test_same_grid_handoff_preserves_gas_mass(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.010)
        state = solver.case_b_initial_state()
        area, discharge, gas_mass, gas_momentum = (
            _map_external_horizontal_state(
                solver,
                state,
                x_target=solver.x,
                full_area=solver.section.full_area,
                dx=solver.dx,
            )
        )
        self.assertAlmostEqual(float(np.sum(gas_mass)), state.gas.mass)
        self.assertTrue(np.all(gas_mass[solver.x > solver.config.valve_x] == 0.0))
        self.assertTrue(np.all(np.isfinite(area)))
        self.assertTrue(np.all(np.isfinite(discharge)))
        self.assertTrue(np.all(np.isfinite(gas_momentum)))

    def test_side_t_face_fluxes_preserve_horizontal_volume_change(self) -> None:
        solver = build_case_a_shockfit_solver(dx=0.040)
        state = solver.case_b_initial_state()
        dt = 2.0e-4
        west_flow = 1.8e-4
        east_flow = 1.4e-4
        expected_change = (east_flow - west_flow) * dt
        initial_volume = float(np.sum(state.area) * solver.dx)
        advanced = solver.apply_junction_liquid_fluxes(
            state,
            west_flow=west_flow,
            east_flow=east_flow,
            dt=dt,
        )
        self.assertAlmostEqual(
            float(np.sum(advanced.area) * solver.dx),
            initial_volume + expected_change,
            places=14,
        )
        self.assertEqual(advanced.gas.mass, state.gas.mass)

    def test_network_boundary_flux_is_exactly_conservative(self) -> None:
        """A T-node flux must enter the FV balance once and only once."""

        section = CASE_A_SHOCKFIT_CORE.CircularSection(
            0.094, wave_speed=28.0
        )
        ncell = 32
        dx = 0.02
        dt = 2.0e-4
        depth = 0.080
        area_value = float(section.area_from_depth(depth))
        area = np.full(ncell, area_value)
        discharge = np.zeros(ncell)
        outward_flow = 2.0e-5
        momentum_flux = (
            outward_flow**2 / area_value
            + float(section.pressure_flux(area_value))
        )
        initial_volume = float(np.sum(area) * dx)
        advanced = CASE_A_SHOCKFIT_CORE.central_upwind_wet_dry_step(
            CASE_A_SHOCKFIT_CORE.WetDryState(area, discharge),
            dx=dx,
            dt=dt,
            section=section,
            left_boundary="wall",
            right_face_flux=(outward_flow, momentum_flux),
        )
        self.assertAlmostEqual(
            float(np.sum(advanced.area) * dx),
            initial_volume - outward_flow * dt,
            places=14,
        )

    def test_muscl_ssprk2_retains_a_smooth_standing_wave(self) -> None:
        """The raw FV field must carry a smooth wave without plot filtering."""

        section = CASE_A_SHOCKFIT_CORE.CircularSection(
            0.094, wave_speed=28.0
        )
        length = 1.0
        still_depth = 0.078
        amplitude = 2.0e-4
        celerity = float(
            section.free_surface_celerity_from_depth(still_depth)
        )
        period = length / celerity
        errors: list[float] = []
        retained: list[float] = []
        volume_errors: list[float] = []

        for ncell in (25, 50, 100):
            dx = length / ncell
            x = (np.arange(ncell, dtype=float) + 0.5) * dx
            initial_depth = still_depth + amplitude * np.cos(
                2.0 * np.pi * x / length
            )
            initial_area = np.asarray(
                section.area_from_depth(initial_depth)
            )
            state = CASE_A_SHOCKFIT_CORE.WetDryState(
                initial_area, np.zeros(ncell)
            )
            elapsed = 0.0
            while elapsed < period - 1.0e-14:
                # Use the same acoustic time scale as the coupled Case-A core.
                dt = min(0.90 * dx / section.wave_speed, period - elapsed)
                state = CASE_A_SHOCKFIT_CORE.central_upwind_wet_dry_step(
                    state,
                    dx=dx,
                    dt=dt,
                    section=section,
                    cfl=0.90,
                )
                elapsed += dt

            depth = np.asarray(section.depth_from_area(state.area))
            errors.append(float(np.mean(np.abs(depth - initial_depth))))
            mode = 2.0 * abs(np.mean(
                (depth - still_depth)
                * np.exp(-2.0j * np.pi * x / length)
            ))
            retained.append(float(mode / amplitude))
            volume_errors.append(float(
                (np.sum(state.area) - np.sum(initial_area)) * dx
            ))

        order_coarse = np.log(errors[0] / errors[1]) / np.log(2.0)
        order_fine = np.log(errors[1] / errors[2]) / np.log(2.0)
        self.assertGreater(order_coarse, 1.8)
        self.assertGreater(order_fine, 1.8)
        self.assertGreater(retained[1], 0.95)
        self.assertLess(max(abs(value) for value in volume_errors), 2.0e-14)

    def test_muscl_wet_dry_step_is_positive_and_mass_conserving(self) -> None:
        section = CASE_A_SHOCKFIT_CORE.CircularSection(
            0.094, wave_speed=28.0
        )
        length = 1.0
        ncell = 100
        dx = length / ncell
        x = (np.arange(ncell, dtype=float) + 0.5) * dx
        reservoir_area = float(section.area_from_depth(0.055))
        area = np.where(x < 0.40, reservoir_area, 0.0)
        state = CASE_A_SHOCKFIT_CORE.WetDryState(
            area, np.zeros_like(area)
        )
        initial_volume = float(np.sum(state.area) * dx)
        elapsed = 0.0
        end_time = 0.15
        dry_area = 1.0e-10 * section.full_area
        while elapsed < end_time - 1.0e-14:
            velocity = np.divide(
                state.discharge,
                state.area,
                out=np.zeros_like(state.discharge),
                where=state.area > dry_area,
            )
            speed = max(float(np.max(
                np.abs(velocity) + np.asarray(section.celerity(state.area))
            )), 1.0e-12)
            dt = min(0.35 * dx / speed, end_time - elapsed)
            state = CASE_A_SHOCKFIT_CORE.central_upwind_wet_dry_step(
                state,
                dx=dx,
                dt=dt,
                section=section,
                cfl=0.45,
            )
            self.assertTrue(np.all(state.area >= 0.0))
            self.assertTrue(np.all(np.isfinite(state.area)))
            self.assertTrue(np.all(np.isfinite(state.discharge)))
            elapsed += dt

        self.assertAlmostEqual(
            float(np.sum(state.area) * dx), initial_volume, places=11
        )
        # The finite-speed wetting front cannot seed a numerical film at the
        # remote closed end over this short interval.
        self.assertTrue(np.all(state.area[x > 0.95] == 0.0))

    def test_draining_limiter_scales_physical_boundary_outflow(self) -> None:
        mass_flux = np.array([-2.0, 0.0, 2.0])
        momentum_flux = np.array([-6.0, 0.0, 6.0])
        area = np.ones(2)
        limited_mass, limited_momentum = (
            CASE_A_SHOCKFIT_CORE._apply_donor_draining_limiter(
                mass_flux,
                momentum_flux,
                area,
                dx=1.0,
                dt=1.0,
            )
        )
        np.testing.assert_allclose(limited_mass, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(limited_momentum, [-3.0, 0.0, 3.0])

if __name__ == "__main__":
    unittest.main()
