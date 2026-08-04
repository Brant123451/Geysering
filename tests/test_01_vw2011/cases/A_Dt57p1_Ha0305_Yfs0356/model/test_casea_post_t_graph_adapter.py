"""Checks for the conservative west-arm/T-graph finite-volume adapter."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from casea_post_t_graph_adapter import (  # noqa: E402
    advance_west_branch,
    west_branch_characteristic,
)
from tosan2021_horizontal_shockfit import (  # noqa: E402
    CircularSection,
    WetDryState,
)


class CaseAPostTGraphAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.section = CircularSection(0.094, wave_speed=28.0)
        self.dx = 0.02
        self.depth = 0.080
        self.area = float(self.section.area_from_depth(self.depth))

    def test_characteristic_reproduces_the_resolved_trace(self) -> None:
        discharge = 1.2e-4
        state = WetDryState(
            np.full(16, self.area),
            np.full(16, discharge),
        )
        pressure = 103_000.0
        characteristic, area = west_branch_characteristic(
            state,
            section=self.section,
            gas_pressure_abs=pressure,
            liquid_density=998.0,
            loss_coefficient=0.0,
        )
        velocity = characteristic.outward_velocity(
            pressure, liquid_density=998.0
        )
        self.assertAlmostEqual(area, self.area)
        self.assertAlmostEqual(velocity, -discharge / self.area, places=14)

    def test_node_flux_changes_west_volume_once(self) -> None:
        state = WetDryState(
            np.full(32, self.area),
            np.zeros(32),
        )
        flow_to_west = 2.0e-5
        result = advance_west_branch(
            state,
            outward_west_liquid_flow=flow_to_west,
            node_liquid_area=self.area,
            dx=self.dx,
            dt=2.0e-4,
            section=self.section,
        )
        self.assertAlmostEqual(
            result.liquid_volume_change,
            flow_to_west * 2.0e-4,
            places=14,
        )
        self.assertAlmostEqual(result.conservation_error, 0.0, places=14)

    def test_uniform_equilibrium_has_no_two_cell_mode(self) -> None:
        state = WetDryState(
            np.full(64, self.area),
            np.zeros(64),
        )
        for _ in range(200):
            result = advance_west_branch(
                state,
                outward_west_liquid_flow=0.0,
                node_liquid_area=self.area,
                dx=self.dx,
                dt=2.0e-4,
                section=self.section,
            )
            state = result.state
        np.testing.assert_allclose(
            state.area,
            self.area,
            rtol=0.0,
            atol=2.0e-14 * self.section.full_area,
        )
        np.testing.assert_allclose(state.discharge, 0.0, rtol=0.0, atol=1.0e-14)


if __name__ == "__main__":
    unittest.main()
