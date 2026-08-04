from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from casea_horizontal_liquid_operator import (  # noqa: E402
    HorizontalLiquidParameters,
    PressurePotentialState,
    pressure_potential_state,
)
from casea_post_t_liquid_stage import (  # noqa: E402
    BranchPressureEvaluation,
    PostTLiquidGeometry,
    advance_post_t_liquid_ssprk2,
    post_t_liquid_stage_rhs,
)


class CaseAPostTLiquidStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.diameter = 0.094
        self.area_full = 0.25 * np.pi * self.diameter**2
        self.dx = 0.02
        self.rho_l = 998.2
        self.p_atm = 101_325.0
        self.params = HorizontalLiquidParameters(
            area_full=self.area_full,
            diameter=self.diameter,
            wave_speed=28.0,
            cell_width=self.dx,
            rho_liquid=self.rho_l,
        )
        self.geometry = PostTLiquidGeometry(
            junction_face_index=24,
            horizontal_cell_width=self.dx,
            vertical_cell_width=self.dx,
            liquid_density=self.rho_l,
        )

    def _gas_mass(self, area: np.ndarray) -> np.ndarray:
        rho = self.p_atm / (
            self.params.gas_constant * self.params.gas_temperature
        )
        return rho * (self.area_full - area) * self.dx

    def _callback(self, equilibrium_area: float):
        equilibrium_mass = float(self._gas_mass(np.array([equilibrium_area]))[0])
        equilibrium_pressure = pressure_potential_state(
            equilibrium_area,
            0.0,
            equilibrium_mass,
            0.0,
            True,
            self.params,
        )
        psi0 = float(equilibrium_pressure.potential)

        def callback(branch, area, discharge, gas_mass, gas_momentum):
            pressure = pressure_potential_state(
                area,
                discharge,
                gas_mass,
                gas_momentum,
                np.ones_like(area, dtype=bool),
                self.params,
            )
            # Convert the conservative potential change to the same local
            # absolute-pressure datum used by the node characteristic.
            face_pressure = self.p_atm + self.rho_l * (
                pressure.potential - psi0
            ) / area
            return BranchPressureEvaluation(
                pressure=pressure,
                face_pressure_abs=np.asarray(face_pressure),
                node_pressure_offset=np.zeros_like(area),
                momentum_source=np.zeros_like(area),
            )

        return callback

    def _uniform_state(self, area_fraction=0.68):
        area = area_fraction * self.area_full
        ah = np.full(48, area)
        qh = np.zeros_like(ah)
        av = np.full(18, area)
        qv = np.zeros_like(av)
        mgh = self._gas_mass(ah)
        mgv = self._gas_mass(av)
        return area, ah, qh, av, qv, mgh, np.zeros_like(mgh), mgv, np.zeros_like(mgv)

    @staticmethod
    def _synthetic_pressure_state(
        area: np.ndarray,
        discharge: np.ndarray,
        potential: np.ndarray,
        celerity: float,
    ) -> PressurePotentialState:
        speed = np.full_like(area, celerity)
        velocity = discharge / area
        return PressurePotentialState(
            potential=np.asarray(potential, dtype=float),
            derivative=speed**2,
            discharge_derivative=np.zeros_like(area),
            celerity=speed,
            eigenvalue_minus=velocity - speed,
            eigenvalue_plus=velocity + speed,
            lambda_value=np.zeros_like(area),
            lambda_derivative=np.zeros_like(area),
            stratified=np.ones_like(area, dtype=bool),
        )

    def test_static_equilibrium_has_zero_rhs_and_no_momentum_pulse(self) -> None:
        area, ah, qh, av, qv, mgh, jgh, mgv, jgv = self._uniform_state()
        result = post_t_liquid_stage_rhs(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            geometry=self.geometry,
            pressure_callback=self._callback(area),
        )
        np.testing.assert_allclose(result.rhs_horizontal_area, 0.0, atol=2e-13)
        np.testing.assert_allclose(
            result.rhs_horizontal_discharge, 0.0, atol=2e-13
        )
        np.testing.assert_allclose(result.rhs_vertical_area, 0.0, atol=2e-13)
        np.testing.assert_allclose(
            result.rhs_vertical_discharge, 0.0, atol=2e-13
        )
        self.assertLess(abs(result.diagnostics.node_volume_residual), 1e-12)
        for change in result.diagnostics.momentum_flux_changes.values():
            self.assertLess(abs(change), 2e-14)

    def test_arbitrary_signed_node_flows_conserve_total_liquid_volume(self) -> None:
        area, ah, qh, av, qv, mgh, jgh, mgv, jgv = self._uniform_state()
        ah[:] = area
        av[:] = area
        # These are independent resolved branch traces, not an imposed split.
        qh[self.geometry.junction_face_index - 1] = 0.18 * area
        qh[self.geometry.junction_face_index] = 0.04 * area
        qv[0] = -0.03 * area
        result = post_t_liquid_stage_rhs(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            geometry=self.geometry,
            pressure_callback=self._callback(area),
        )
        flows = result.diagnostics.solution.branch_fluxes
        signed = [flows[name].volume_flux for name in ("west", "east", "vertical")]
        self.assertTrue(any(value < 0.0 for value in signed))
        self.assertTrue(any(value > 0.0 for value in signed))
        integrated = (
            np.sum(result.rhs_horizontal_area) * self.dx
            + np.sum(result.rhs_vertical_area) * self.dx
        )
        self.assertLessEqual(
            abs(integrated), self.geometry.node_volume_flux_tolerance
        )
        self.assertAlmostEqual(sum(signed), 0.0, places=12)
        self.assertAlmostEqual(
            result.diagnostics.coordinate_volume_fluxes["west"],
            -flows["west"].volume_flux,
            places=15,
        )
        self.assertAlmostEqual(
            result.diagnostics.coordinate_volume_fluxes["east"],
            flows["east"].volume_flux,
            places=15,
        )

    def test_ssprk2_does_not_grow_the_two_cell_mode(self) -> None:
        area, ah, qh, av, qv, mgh, jgh, mgv, jgv = self._uniform_state()
        checker_h = (-1.0) ** np.arange(ah.size)
        checker_v = (-1.0) ** np.arange(av.size)
        amplitude = 2.0e-6 * area
        ah += amplitude * checker_h
        av += amplitude * checker_v
        initial = np.sqrt(
            np.mean(((ah - area) * checker_h) ** 2)
            + np.mean(((av - area) * checker_v) ** 2)
        )
        # This test isolates the RK/node discretisation.  Holding a separate
        # compressible-gas mass fixed in every checkerboard cell for many
        # liquid-only steps is not the coupled model and can deliberately cross
        # the IKH neutral point.  Use a constant hyperbolic pressure law here;
        # gas/liquid simultaneous recomputation is covered by the coupled-stage
        # tests.
        def callback(branch, local_area, local_q, gas_mass, gas_momentum):
            local_area = np.asarray(local_area, dtype=float)
            local_q = np.asarray(local_q, dtype=float)
            potential = 28.0**2 * local_area
            pressure = self._synthetic_pressure_state(
                local_area, local_q, potential, 28.0
            )
            return BranchPressureEvaluation(
                pressure=pressure,
                face_pressure_abs=np.full_like(local_area, self.p_atm),
                node_pressure_offset=np.zeros_like(local_area),
                momentum_source=np.zeros_like(local_area),
                potential_pressure_abs=np.full_like(local_area, self.p_atm),
            )
        elapsed = 0.0
        end_time = 0.010
        dt = 0.20 * self.dx / 28.0
        while elapsed < end_time - 1e-15:
            step = min(dt, end_time - elapsed)
            advanced = advance_post_t_liquid_ssprk2(
                ah, qh, av, qv, mgh, jgh, mgv, jgv,
                dt=step,
                geometry=self.geometry,
                pressure_callback=callback,
            )
            ah = advanced.horizontal_area
            qh = advanced.horizontal_discharge
            av = advanced.vertical_area
            qv = advanced.vertical_discharge
            elapsed += step
        final = np.sqrt(
            np.mean(((ah - area) * checker_h) ** 2)
            + np.mean(((av - area) * checker_v) ** 2)
        )
        self.assertLessEqual(final, initial * (1.0 + 1.0e-9))

    def test_ssprk2_reports_machine_closed_volume_ledger(self) -> None:
        area, ah, qh, av, qv, mgh, jgh, mgv, jgv = self._uniform_state()
        qh[:] = 0.01 * area
        qv[:] = -0.005 * area
        advanced = advance_post_t_liquid_ssprk2(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            dt=1.0e-5,
            geometry=self.geometry,
            pressure_callback=self._callback(area),
        )
        self.assertLess(abs(advanced.conservation_error), 2.0e-15)
        self.assertLess(abs(advanced.node_volume_integral), 1.0e-14)

    def test_hydrostatic_column_with_atmospheric_surface_is_well_balanced(self) -> None:
        active_count = 8
        vertical_cells = 14
        area = 0.72 * self.area_full
        ah = np.full(48, area)
        qh = np.zeros_like(ah)
        av = np.zeros(vertical_cells)
        av[:active_count] = area
        qv = np.zeros_like(av)
        mgh = self._gas_mass(ah)
        jgh = np.zeros_like(mgh)
        mgv = np.zeros_like(av)
        jgv = np.zeros_like(av)
        gravity = 9.81
        height = active_count * self.dx
        bottom_pressure = self.p_atm + self.rho_l * gravity * height

        def hydrostatic_callback(branch, local_area, discharge, gas_mass, gas_momentum):
            if branch == "horizontal":
                pressure_abs = np.full_like(local_area, bottom_pressure)
                potential = pressure_abs * local_area / self.rho_l
                source = np.zeros_like(local_area)
                characteristic_pressure = pressure_abs.copy()
            else:
                z = (np.arange(local_area.size) + 0.5) * self.dx
                pressure_abs = self.p_atm + self.rho_l * gravity * (height - z)
                potential = pressure_abs * local_area / self.rho_l
                source = -gravity * local_area
                characteristic_pressure = pressure_abs.copy()
                # The incoming characteristic is referenced at the bottom
                # face, whereas its conservative cell potential is centered.
                characteristic_pressure[0] = bottom_pressure
            return BranchPressureEvaluation(
                pressure=self._synthetic_pressure_state(
                    local_area, discharge, potential, 28.0
                ),
                face_pressure_abs=characteristic_pressure,
                node_pressure_offset=np.zeros_like(local_area),
                momentum_source=source,
                potential_pressure_abs=pressure_abs,
            )

        result = post_t_liquid_stage_rhs(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            geometry=self.geometry,
            pressure_callback=hydrostatic_callback,
            vertical_active_count=active_count,
        )
        np.testing.assert_allclose(result.rhs_horizontal_area, 0.0, atol=2e-13)
        np.testing.assert_allclose(
            result.rhs_horizontal_discharge, 0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            result.rhs_vertical_area[:active_count], 0.0, atol=2e-13
        )
        np.testing.assert_allclose(
            result.rhs_vertical_discharge[:active_count], 0.0, atol=2e-12
        )
        self.assertTrue(
            np.array_equal(
                result.rhs_vertical_area[active_count:],
                np.zeros(vertical_cells - active_count),
            )
        )
        self.assertTrue(
            np.array_equal(
                result.rhs_vertical_discharge[active_count:],
                np.zeros(vertical_cells - active_count),
            )
        )
        self.assertAlmostEqual(
            result.diagnostics.vertical_top_momentum_flux,
            self.p_atm * area / self.rho_l,
            places=14,
        )

    def test_dry_vertical_suffix_is_not_instantaneously_filled(self) -> None:
        active_count = 6
        area, ah, qh, _, _, mgh, jgh, _, _ = self._uniform_state()
        av = np.zeros(12)
        av[:active_count] = area
        qv = np.zeros_like(av)
        mgv = np.zeros_like(av)
        mgv[:active_count] = self._gas_mass(av[:active_count])
        jgv = np.zeros_like(av)
        qv[0] = 0.02 * area
        callback = self._callback(area)
        result = post_t_liquid_stage_rhs(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            geometry=self.geometry,
            pressure_callback=callback,
            vertical_active_mask=np.arange(av.size) < active_count,
        )
        self.assertTrue(
            np.array_equal(
                result.rhs_vertical_area[active_count:],
                np.zeros(av.size - active_count),
            )
        )
        self.assertTrue(
            np.array_equal(
                result.rhs_vertical_discharge[active_count:],
                np.zeros(av.size - active_count),
            )
        )
        advanced = advance_post_t_liquid_ssprk2(
            ah, qh, av, qv, mgh, jgh, mgv, jgv,
            dt=2.0e-6,
            geometry=self.geometry,
            pressure_callback=callback,
            vertical_active_count=active_count,
        )
        self.assertTrue(
            np.array_equal(
                advanced.vertical_area[active_count:],
                np.zeros(av.size - active_count),
            )
        )
        self.assertTrue(
            np.array_equal(
                advanced.vertical_discharge[active_count:],
                np.zeros(av.size - active_count),
            )
        )

    def test_bottom_inflow_and_outflow_change_column_volume_exactly(self) -> None:
        active_count = 7
        for direction in (-1.0, 1.0):
            with self.subTest(direction=direction):
                area, ah, qh, _, _, mgh, jgh, _, _ = self._uniform_state()
                av = np.zeros(13)
                av[:active_count] = area
                qv = np.zeros_like(av)
                qv[0] = direction * 0.06 * area
                mgv = np.zeros_like(av)
                mgv[:active_count] = self._gas_mass(av[:active_count])
                jgv = np.zeros_like(av)
                callback = self._callback(area)
                result = post_t_liquid_stage_rhs(
                    ah, qh, av, qv, mgh, jgh, mgv, jgv,
                    geometry=self.geometry,
                    pressure_callback=callback,
                    vertical_active_count=active_count,
                )
                node_flow = result.diagnostics.coordinate_volume_fluxes["vertical"]
                self.assertGreater(direction * node_flow, 0.0)
                column_rate = float(np.sum(result.rhs_vertical_area) * self.dx)
                self.assertAlmostEqual(column_rate, node_flow, places=14)

                dt = 2.0e-6
                advanced = advance_post_t_liquid_ssprk2(
                    ah, qh, av, qv, mgh, jgh, mgv, jgv,
                    dt=dt,
                    geometry=self.geometry,
                    pressure_callback=callback,
                    vertical_active_count=active_count,
                )
                expected_column_change = 0.5 * dt * (
                    advanced.first_stage.coordinate_volume_fluxes["vertical"]
                    + advanced.second_stage.coordinate_volume_fluxes["vertical"]
                )
                actual_column_change = float(
                    np.sum(advanced.vertical_area - av) * self.dx
                )
                self.assertAlmostEqual(
                    actual_column_change, expected_column_change, places=15
                )
                self.assertLess(abs(advanced.conservation_error), 2.0e-15)


if __name__ == "__main__":
    unittest.main()
