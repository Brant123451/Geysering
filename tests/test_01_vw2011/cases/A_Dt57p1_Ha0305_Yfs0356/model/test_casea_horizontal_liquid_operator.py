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
    _circular_depth_and_width,
    _circular_hydrostatic_state,
    characteristic_spectral_radius,
    decoupled_lambda_and_derivative,
    physical_liquid_flux,
    pressure_potential_state,
    pressure_potential_wave_state,
    rusanov_face_flux,
    ssprk2_stage_step,
)


class HorizontalLiquidOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        diameter = 0.094
        self.params = HorizontalLiquidParameters(
            area_full=0.25 * np.pi * diameter**2,
            diameter=diameter,
            wave_speed=28.0,
            cell_width=0.02,
        )

    def _gas_state(self, area: float, pressure_ratio: float = 1.01):
        gas_area = self.params.area_full - area
        rho = (
            pressure_ratio * self.params.atmospheric_pressure
            / (self.params.gas_constant * self.params.gas_temperature)
        )
        return rho * gas_area * self.params.cell_width, 0.0

    def test_lambda_derivative_matches_finite_difference(self) -> None:
        area = 0.72 * self.params.area_full
        discharge = 0.08 * area
        mass, momentum = self._gas_state(area)
        lam, derivative = decoupled_lambda_and_derivative(
            area, discharge, mass, momentum, self.params
        )
        step = 2.0e-6 * self.params.area_full
        plus, _ = decoupled_lambda_and_derivative(
            area + step, discharge, mass, momentum, self.params
        )
        minus, _ = decoupled_lambda_and_derivative(
            area - step, discharge, mass, momentum, self.params
        )
        finite_difference = (plus - minus) / (2.0 * step)
        self.assertAlmostEqual(
            float(derivative), float(finite_difference),
            delta=2.0e-5 * max(abs(float(finite_difference)), 1.0),
        )
        self.assertTrue(np.isfinite(float(lam)))

    def test_half_full_circular_geometry_uses_exact_endpoint_root(self) -> None:
        area = 0.5 * self.params.area_full
        depth, width = _circular_depth_and_width(
            np.asarray(area), self.params
        )
        self.assertEqual(float(depth), 0.5 * self.params.diameter)
        self.assertEqual(float(width), self.params.diameter)

        potential, _, _, _ = _circular_hydrostatic_state(area, self.params)
        radius = 0.5 * self.params.diameter
        expected = self.params.gravity * (2.0 / 3.0) * radius**3
        self.assertAlmostEqual(float(potential), expected, places=18)

    def test_wave_state_matches_full_pressure_state_for_scalar_topologies(self) -> None:
        area = 0.63 * self.params.area_full
        mass, momentum = self._gas_state(area, pressure_ratio=1.03)
        offset = 2.75e-4
        for supported in (False, True):
            full = pressure_potential_state(
                area,
                -0.17 * area,
                mass,
                momentum,
                supported,
                self.params,
                stratified_potential_offset=offset,
            )
            wave = pressure_potential_wave_state(
                area,
                supported,
                self.params,
                stratified_potential_offset=offset,
            )
            np.testing.assert_array_equal(wave.potential, full.potential)
            np.testing.assert_array_equal(wave.celerity, full.celerity)

    def test_wave_state_matches_full_pressure_state_for_mixed_arrays(self) -> None:
        area = self.params.area_full * np.array(
            [[0.42, 0.76, 1.00], [0.58, 0.94, 1.015]]
        )
        supported = np.array(
            [[True, False, True], [False, True, False]],
            dtype=bool,
        )
        discharge = area * np.array(
            [[-0.20, 0.05, 0.00], [0.13, -0.08, 0.02]]
        )
        gas_area = np.maximum(self.params.area_full - area, 1.0e-8)
        gas_mass = (
            self.params.atmospheric_gas_density
            * gas_area
            * self.params.cell_width
        )
        gas_momentum = gas_mass * np.array(
            [[4.0, -2.0, 0.0], [0.5, 3.0, -1.0]]
        )
        offset = np.array(
            [[1.0e-4, 2.0e-4, 3.0e-4], [4.0e-4, 5.0e-4, 6.0e-4]]
        )

        full = pressure_potential_state(
            area,
            discharge,
            gas_mass,
            gas_momentum,
            supported,
            self.params,
            stratified_potential_offset=offset,
        )
        wave = pressure_potential_wave_state(
            area,
            supported,
            self.params,
            stratified_potential_offset=offset,
        )
        np.testing.assert_array_equal(wave.potential, full.potential)
        np.testing.assert_array_equal(wave.celerity, full.celerity)

    def test_riemann_speed_matches_circular_shallow_water_celerity(self) -> None:
        area = 0.68 * self.params.area_full
        discharge = 0.0
        mass, _ = self._gas_state(area, pressure_ratio=1.0)
        momentum = 0.0
        pressure = pressure_potential_state(
            area, discharge, mass, momentum, True, self.params
        )
        reported = float(characteristic_spectral_radius(
            area, discharge, pressure
        ))
        published = float(np.sqrt(
            pressure.lambda_value * area
            + self.params.numerical_celerity_floor**2
        ))
        self.assertAlmostEqual(reported, published, places=12)
        self.assertAlmostEqual(float(pressure.celerity), published, places=12)

        face_flux, face_speed = rusanov_face_flux(
            area, discharge, pressure,
            1.001 * area, 0.0,
            pressure_potential_state(
                1.001 * area, 0.0,
                mass, momentum, True, self.params,
            ),
        )
        self.assertTrue(np.all(np.isfinite(face_flux)))
        self.assertGreaterEqual(float(face_speed), reported)

    def test_crown_pressure_potential_is_continuous(self) -> None:
        transition = self.params.elastic_separation_area
        below_area = np.nextafter(transition, 0.0)
        epsilon = transition - below_area
        mass, momentum = self._gas_state(below_area)
        below_natural = pressure_potential_state(
            below_area, 0.0, mass, momentum, True, self.params
        )
        at = pressure_potential_state(
            transition, 0.0, mass, momentum, True, self.params
        )
        offset = float(at.potential - below_natural.potential)
        below = pressure_potential_state(
            below_area,
            0.0,
            mass,
            momentum,
            True,
            self.params,
            stratified_potential_offset=offset,
        )
        above = pressure_potential_state(
            transition + epsilon, 0.0, mass, momentum, True, self.params
        )
        scale = max(abs(float(at.potential)), 1.0e-12)
        self.assertLess(abs(float(below.potential - at.potential)), 2.0e-5 * scale)
        self.assertLess(abs(float(above.potential - at.potential)), 2.0e-5 * scale)

    def test_gas_liquid_slip_does_not_enter_shallow_water_pressure(self) -> None:
        area = 0.55 * self.params.area_full
        mass, _ = self._gas_state(area)
        high_slip = pressure_potential_state(
            area, -0.2 * area, mass, mass * 200.0, True, self.params
        )
        zero_slip = pressure_potential_state(
            area, -0.2 * area, mass, -0.2 * mass, True, self.params
        )
        self.assertAlmostEqual(
            float(high_slip.potential), float(zero_slip.potential), places=14
        )
        self.assertAlmostEqual(
            float(high_slip.celerity), float(zero_slip.celerity), places=14
        )
        self.assertGreater(float(high_slip.lambda_value), 0.0)

    def test_all_holdups_use_the_same_shallow_water_slip_independent_law(self) -> None:
        area = 0.90 * self.params.area_full
        discharge = -0.2 * area
        mass, _ = self._gas_state(area, pressure_ratio=1.02)
        high_slip, _ = decoupled_lambda_and_derivative(
            area, discharge, mass, mass * 200.0, self.params
        )
        zero_slip, _ = decoupled_lambda_and_derivative(
            area, discharge, mass, mass * (discharge / area), self.params
        )
        self.assertAlmostEqual(float(high_slip), float(zero_slip), places=12)
        self.assertGreater(float(high_slip), 0.0)

    def test_vent_connected_underpressure_is_not_negative_liquid_stiffness(self) -> None:
        area = 0.70 * self.params.area_full
        discharge = 0.0
        mass, _ = self._gas_state(area, pressure_ratio=0.25)
        lam, _ = decoupled_lambda_and_derivative(
            area, discharge, mass, 0.0, self.params
        )
        self.assertGreater(float(lam), 0.0)

    def test_ssprk2_recomputes_each_stage_and_does_not_grow_2dx_mode(self) -> None:
        ncell = 64
        dx = 1.0 / ncell
        x = (np.arange(ncell) + 0.5) * dx
        still_area = 0.60 * self.params.area_full
        area = still_area * (
            1.0 + 2.0e-3 * np.sin(2.0 * np.pi * x)
            + 2.0e-5 * (-1.0) ** np.arange(ncell)
        )
        discharge = np.zeros(ncell)
        constant_c = 0.8
        calls: list[tuple[float, float]] = []

        def pressure_for(
            local_area: np.ndarray, local_q: np.ndarray
        ) -> PressurePotentialState:
            potential = 0.5 * constant_c**2 * local_area**2 / still_area
            derivative = constant_c**2 * local_area / still_area
            return PressurePotentialState(
                potential=potential,
                derivative=derivative,
                discharge_derivative=np.zeros_like(local_area),
                celerity=np.sqrt(derivative),
                eigenvalue_minus=local_q / local_area - np.sqrt(derivative),
                eigenvalue_plus=local_q / local_area + np.sqrt(derivative),
                lambda_value=np.zeros_like(local_area),
                lambda_derivative=np.zeros_like(local_area),
                stratified=np.ones_like(local_area, dtype=bool),
            )

        def rhs(local_area: np.ndarray, local_q: np.ndarray, stage_time: float):
            calls.append((stage_time, float(np.linalg.norm(local_q))))
            left_pressure = pressure_for(local_area, local_q)
            right_area = np.roll(local_area, -1)
            right_q = np.roll(local_q, -1)
            right_pressure = pressure_for(right_area, right_q)
            flux, _ = rusanov_face_flux(
                local_area, local_q, left_pressure,
                right_area, right_q, right_pressure,
            )
            divergence = (flux - np.roll(flux, 1, axis=0)) / dx
            return -divergence[:, 0], -divergence[:, 1]

        checker = (-1.0) ** np.arange(ncell)
        initial_checker = abs(float(np.mean((area - still_area) * checker)))
        elapsed = 0.0
        end_time = 0.20
        dt_nominal = 0.35 * dx / constant_c
        while elapsed < end_time - 1.0e-14:
            dt = min(dt_nominal, end_time - elapsed)
            area, discharge = ssprk2_stage_step(
                area, discharge, elapsed, dt, rhs
            )
            elapsed += dt
        final_checker = abs(float(np.mean((area - still_area) * checker)))
        self.assertLessEqual(final_checker, initial_checker * (1.0 + 1.0e-10))
        self.assertEqual(len(calls) % 2, 0)
        self.assertGreater(calls[1][1], calls[0][1])
        self.assertGreater(calls[1][0], calls[0][0])


if __name__ == "__main__":
    unittest.main()
