from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from casea_horizontal_liquid_operator import (  # noqa: E402
    HorizontalLiquidParameters,
    physical_liquid_flux,
)
from casea_post_t_liquid_stage import (  # noqa: E402
    PostTLiquidGeometry,
    post_t_liquid_stage_rhs,
)
from casea_post_t_physical_closure import (  # noqa: E402
    CaseAPostTClosureParameters,
    CaseAPostTPhysicalClosure,
    physical_momentum_source,
)


class CaseAPostTPhysicalClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rho_l = 998.0
        self.p_atm = 101_325.0
        self.gravity = 9.81
        self.dx = 0.02
        self.diameter = 0.094
        self.area = 0.25 * np.pi * self.diameter**2
        common = dict(
            area_full=self.area,
            diameter=self.diameter,
            wave_speed=28.0,
            cell_width=self.dx,
            gravity=self.gravity,
            rho_liquid=self.rho_l,
            gas_temperature=293.0,
            atmospheric_pressure=self.p_atm,
        )
        self.horizontal = HorizontalLiquidParameters(**common)
        self.vertical = HorizontalLiquidParameters(**common, tension_head=0.0)
        self.parameters = CaseAPostTClosureParameters(
            horizontal=self.horizontal,
            vertical=self.vertical,
            vertical_cell_width=self.dx,
        )
        self.closure = CaseAPostTPhysicalClosure(self.parameters)
        self.rho_atm = self.horizontal.atmospheric_gas_density

    def test_static_gas_elastic_contact_has_no_pressure_impulse(self) -> None:
        area = np.array(
            [0.68 * self.area, 0.68 * self.area, self.area, self.area]
        )
        q = np.zeros_like(area)
        void = self.area - area
        mass = self.rho_atm * void * self.dx
        momentum = np.zeros_like(area)
        evaluation = self.closure("horizontal", area, q, mass, momentum)

        np.testing.assert_allclose(
            evaluation.face_pressure_abs, self.p_atm, rtol=0.0, atol=2e-10
        )
        flux = physical_liquid_flux(area, q, evaluation.pressure)
        # The connected gas component receives one gauge, matched to the
        # adjacent elastic trace; all static momentum tractions are identical.
        np.testing.assert_allclose(flux[:, 1], flux[-1, 1], atol=2e-13)
        self.assertTrue(np.all(evaluation.pressure.celerity > 0.0))

    def test_atmospheric_column_is_exactly_well_balanced(self) -> None:
        active_count = 5
        horizontal_cells = 8
        vertical_cells = 9
        height = active_count * self.dx
        # Elastic horizontal area represents the same bottom pressure as the
        # atmospheric vertical hydrostatic column.
        area_h = self.area * (
            1.0 + self.gravity * height / self.horizontal.wave_speed**2
        )
        ah = np.full(horizontal_cells, area_h)
        qh = np.zeros_like(ah)
        av = np.zeros(vertical_cells)
        av[:active_count] = self.area
        qv = np.zeros_like(av)
        mgh = np.zeros_like(ah)
        jgh = np.zeros_like(ah)
        mgv = np.zeros_like(av)
        jgv = np.zeros_like(av)
        geometry = PostTLiquidGeometry(
            junction_face_index=4,
            horizontal_cell_width=self.dx,
            vertical_cell_width=self.dx,
            liquid_density=self.rho_l,
            atmospheric_pressure_abs=self.p_atm,
        )
        result = post_t_liquid_stage_rhs(
            ah,
            qh,
            av,
            qv,
            mgh,
            jgh,
            mgv,
            jgv,
            geometry=geometry,
            pressure_callback=self.closure,
            vertical_active_count=active_count,
        )
        np.testing.assert_allclose(result.rhs_horizontal_area, 0.0, atol=2e-13)
        np.testing.assert_allclose(
            result.rhs_horizontal_discharge, 0.0, atol=3e-11
        )
        np.testing.assert_allclose(
            result.rhs_vertical_area[:active_count], 0.0, atol=1e-11
        )
        np.testing.assert_allclose(
            result.rhs_vertical_discharge[:active_count], 0.0, atol=3e-11
        )
        self.assertTrue(
            np.array_equal(
                result.rhs_vertical_discharge[active_count:],
                np.zeros(vertical_cells - active_count),
            )
        )

    def test_gravity_and_wall_friction_have_physical_signs(self) -> None:
        area = np.full(3, self.area)
        q_positive = np.full(3, 0.04 * self.area)
        q_negative = -q_positive
        horizontal_positive = physical_momentum_source(
            "horizontal", area, q_positive, self.parameters
        )
        horizontal_negative = physical_momentum_source(
            "horizontal", area, q_negative, self.parameters
        )
        self.assertTrue(np.all(horizontal_positive < 0.0))
        self.assertTrue(np.all(horizontal_negative > 0.0))
        np.testing.assert_allclose(
            horizontal_positive, -horizontal_negative, rtol=1e-14, atol=0.0
        )

        vertical_rest = physical_momentum_source(
            "vertical", area, np.zeros_like(area), self.parameters
        )
        np.testing.assert_allclose(vertical_rest, -self.gravity * area)
        vertical_up = physical_momentum_source(
            "vertical", area, q_positive, self.parameters
        )
        vertical_down = physical_momentum_source(
            "vertical", area, q_negative, self.parameters
        )
        self.assertTrue(np.all(vertical_up < vertical_rest))
        self.assertTrue(np.all(vertical_down > vertical_rest))

    def test_mild_two_phase_states_are_finite_and_positive(self) -> None:
        for branch, local in (
            ("horizontal", self.horizontal),
            ("vertical", self.vertical),
        ):
            with self.subTest(branch=branch):
                area = local.area_full * np.array([0.60, 0.67, 0.73, 0.81])
                q = area * np.array([0.02, -0.01, 0.015, 0.0])
                void = local.area_full - area
                density = self.rho_atm * np.array([1.02, 1.05, 1.03, 1.01])
                mass = density * void * local.cell_width
                gas_velocity = np.array([0.03, 0.01, -0.01, 0.0])
                momentum = mass * gas_velocity
                evaluation = self.closure(branch, area, q, mass, momentum)
                for values in (
                    evaluation.pressure.potential,
                    evaluation.pressure.derivative,
                    evaluation.pressure.celerity,
                    evaluation.face_pressure_abs,
                    evaluation.momentum_source,
                ):
                    self.assertTrue(np.all(np.isfinite(values)))
                self.assertTrue(np.all(evaluation.pressure.derivative > 0.0))
                self.assertTrue(np.all(evaluation.pressure.celerity > 0.0))
                self.assertTrue(np.all(evaluation.face_pressure_abs > 0.0))

    def test_vertical_theta_90_removes_cross_section_gravity_wave(self) -> None:
        area = np.full(3, 0.70 * self.area)
        q = np.zeros_like(area)
        void = self.area - area
        mass = self.rho_atm * void * self.dx
        momentum = np.zeros_like(area)
        horizontal = self.closure("horizontal", area, q, mass, momentum)
        vertical = self.closure("vertical", area, q, mass, momentum)
        self.assertTrue(np.all(horizontal.pressure.lambda_value > 0.0))
        np.testing.assert_allclose(
            vertical.pressure.lambda_value, 0.0, rtol=0.0, atol=1e-10
        )
        np.testing.assert_allclose(
            vertical.pressure.celerity,
            self.vertical.numerical_celerity_floor,
            rtol=0.0,
            atol=2e-10,
        )


if __name__ == "__main__":
    unittest.main()
