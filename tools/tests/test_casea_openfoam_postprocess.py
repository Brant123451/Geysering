import importlib.util
import unittest
from pathlib import Path

import numpy as np


class CaseAOpenFOAMPostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.case = (
            root
            / "tests"
            / "test_01_vw2011"
            / "cases"
            / "A_Dt57p1_Ha0305_Yfs0356"
        )
        script = cls.case / "openfoam" / "2d" / "postprocess_compare.py"
        spec = importlib.util.spec_from_file_location("casea_postprocess", script)
        assert spec is not None and spec.loader is not None
        cls.postprocess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.postprocess)

    def test_connected_level_ignores_detached_lower_water(self) -> None:
        y = np.array([0.052, 0.062, 0.072, 0.082, 0.092])
        alpha = np.array([[0.95, 0.10, 0.95, 0.95, 0.10]])

        disconnected_yint, _ = self.postprocess.extract_levels(
            alpha,
            y_locations=y,
            interface_threshold=0.90,
            connected_to_free_surface=False,
        )
        connected_yint, connected_yfs = self.postprocess.extract_levels(
            alpha,
            y_locations=y,
            interface_threshold=0.90,
            free_surface_threshold=0.50,
            connected_to_free_surface=True,
        )

        self.assertEqual(disconnected_yint[0], 0.0)
        self.assertGreater(connected_yint[0], 0.03)
        self.assertGreater(connected_yfs[0], connected_yint[0])

    def test_interface_and_free_surface_thresholds_are_independent(self) -> None:
        y = np.array([0.052, 0.062, 0.072, 0.082])
        alpha = np.array([[0.20, 0.65, 0.70, 0.10]])
        yint, yfs = self.postprocess.extract_levels(
            alpha,
            y_locations=y,
            interface_threshold=0.90,
            free_surface_threshold=0.50,
            connected_to_free_surface=True,
        )

        self.assertTrue(np.isnan(yint[0]))
        self.assertTrue(np.isfinite(yfs[0]))

    def test_climb_fit_stops_at_physical_catch(self) -> None:
        time = np.array([7.4, 7.5, 7.6, 7.7, 7.8, 7.9])
        yint = np.array([0.00, 0.10, 0.20, 0.30, 0.59, 0.20])
        yfs = np.array([0.57, 0.58, 0.59, 0.60, 0.62, 0.25])

        result = self.postprocess.analyse_interface_trajectory(time, yint, yfs)

        self.assertAlmostEqual(result["catch_Tstar"], 7.8)
        self.assertAlmostEqual(result["climb_velocity_Vstar"], 1.0)
        self.assertAlmostEqual(result["climb_fit_R_squared"], 1.0)
        self.assertEqual(result["climb_fit_samples"], 3)


if __name__ == "__main__":
    unittest.main()
