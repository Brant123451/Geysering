"""Control-volume checks for the Case-A side-T liquid adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_shockfit_network import build_case_a_shockfit_solver  # noqa: E402


class CaseATJunctionControlVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = build_case_a_shockfit_solver(dx=0.040)
        self.state = self.solver.case_b_initial_state()
        self.face = self.solver.junction_face_index
        self.cells = np.asarray([self.face - 1, self.face], dtype=int)

    def test_vertical_return_adds_no_axial_momentum(self) -> None:
        discharge = np.asarray(self.state.discharge, dtype=float).copy()
        discharge[self.cells] = np.asarray([1.4e-4, -0.9e-4])
        state = replace(self.state, discharge=discharge)
        area_before = state.area[self.cells].copy()
        discharge_before = state.discharge[self.cells].copy()

        advanced = self.solver.apply_junction_liquid_fluxes(
            state,
            west_flow=-1.0e-4,
            east_flow=1.0e-4,
            dt=1.0e-4,
        )

        self.assertTrue(np.all(advanced.area[self.cells] > area_before))
        np.testing.assert_allclose(
            advanced.discharge[self.cells],
            discharge_before,
            rtol=0.0,
            atol=0.0,
        )

    def test_vertical_withdrawal_carries_local_axial_velocity(self) -> None:
        discharge = np.asarray(self.state.discharge, dtype=float).copy()
        discharge[self.cells] = np.asarray([1.4e-4, -0.9e-4])
        state = replace(self.state, discharge=discharge)
        velocity_before = (
            state.discharge[self.cells] / state.area[self.cells]
        )

        advanced = self.solver.apply_junction_liquid_fluxes(
            state,
            west_flow=1.0e-4,
            east_flow=-1.0e-4,
            dt=1.0e-4,
        )

        self.assertTrue(np.all(advanced.area[self.cells] < state.area[self.cells]))
        np.testing.assert_allclose(
            advanced.discharge[self.cells] / advanced.area[self.cells],
            velocity_before,
            rtol=0.0,
            atol=2.0e-15,
        )


if __name__ == "__main__":
    unittest.main()
