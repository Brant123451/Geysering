from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from casea_postarrival_twofluid import (  # noqa: E402
    HorizontalGasParameters,
    advance_horizontal_gas,
)


class PostArrivalHorizontalGasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = HorizontalGasParameters(diameter=0.094)
        self.n = 80
        self.dx = 0.01
        self.x = (np.arange(self.n) + 0.5) * self.dx
        self.area = self.params.area_full

    def _atmospheric_state(self, liquid_area: np.ndarray):
        gas_area = np.maximum(
            self.area - liquid_area,
            self.params.void_floor_fraction * self.area,
        )
        mass = self.params.rho_atmospheric * gas_area * self.dx
        return mass, np.zeros_like(mass)

    def test_nonuniform_area_atmosphere_remains_at_rest(self) -> None:
        # Switch off the separate interface-elevation gravity term to isolate
        # the quasi-1D nozzle balance tested here.
        params = HorizontalGasParameters(diameter=0.094, gravity=0.0)
        alpha_l = np.linspace(0.15, 0.98, self.n)
        liquid_area = alpha_l * self.area
        mass, momentum = self._atmospheric_state(liquid_area)
        result = advance_horizontal_gas(
            mass,
            momentum,
            liquid_area,
            np.zeros(self.n),
            self.x,
            self.dx,
            0.01,
            self.n - 1,
            params,
        )
        np.testing.assert_allclose(result.mass, mass, rtol=0.0, atol=2.0e-14)
        np.testing.assert_allclose(result.momentum, 0.0, rtol=0.0, atol=2.0e-12)
        self.assertLess(abs(result.mass_error), 2.0e-14)

    def test_pressure_gradient_generates_motion_and_conserves_mass(self) -> None:
        liquid_area = np.full(self.n, 0.55 * self.area)
        mass, momentum = self._atmospheric_state(liquid_area)
        mass[: self.n // 2] *= 1.01
        result = advance_horizontal_gas(
            mass,
            momentum,
            liquid_area,
            np.zeros(self.n),
            self.x,
            self.dx,
            0.002,
            self.n - 1,
            self.params,
        )
        self.assertLess(abs(result.mass_error), 5.0e-13)
        self.assertGreater(float(np.max(np.abs(result.momentum))), 1.0e-9)
        self.assertGreater(result.kinetic_energy, 0.0)
        self.assertGreater(result.substeps, 1)

    def test_only_west_branch_is_advanced(self) -> None:
        junction = 49
        liquid_area = np.full(self.n, 0.60 * self.area)
        mass, momentum = self._atmospheric_state(liquid_area)
        momentum[junction + 1 :] = 7.0e-6
        original_east_mass = mass[junction + 1 :].copy()
        original_east_momentum = momentum[junction + 1 :].copy()
        result = advance_horizontal_gas(
            mass,
            momentum,
            liquid_area,
            np.zeros(self.n),
            self.x,
            self.dx,
            0.001,
            junction,
            self.params,
        )
        np.testing.assert_array_equal(result.mass[junction + 1 :], original_east_mass)
        np.testing.assert_array_equal(
            result.momentum[junction + 1 :], original_east_momentum
        )


if __name__ == "__main__":
    unittest.main()
