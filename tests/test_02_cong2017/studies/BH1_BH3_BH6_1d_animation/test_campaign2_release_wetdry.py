from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_ROOT = HERE.parents[1]
CASES = TEST_ROOT / "cases"
CASE_IDS = (
    "BH1_Dr16_H066_L061",
    "BH3_Dr26_H066_L061",
    "BH6_Dr41_H066_L061",
)
MODEL_FILES = tuple(
    CASES / case_id / "model" / "cong2017_network_twofluid.py"
    for case_id in CASE_IDS
)


def load_model():
    spec = importlib.util.spec_from_file_location(
        "cong2017_network_twofluid_wetdry_test", MODEL_FILES[1]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_FILES[1]}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Campaign2ReleaseWetDryTests(unittest.TestCase):
    def test_three_selected_cases_use_one_identical_core(self) -> None:
        digests = {
            hashlib.sha256(
                path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
            ).hexdigest()
            for path in MODEL_FILES
        }
        self.assertEqual(len(digests), 1)

    def test_true_dry_reach_and_finite_wetting_front(self) -> None:
        model = load_model()
        case = model.NetworkCase(
            D=0.050,
            Dr=0.026,
            riser_height=1.80,
            L_up=3.47,
            L_mid=2.51,
            L_down=0.61,
            x_riser_at=3.47,
            pocket_downstream=True,
            reservoir_head=0.61,
            air_head=0.0,
            init_water_level=0.61,
            Hop_cap=10.0,
            x_transducer_at=6.44,
            valve_open_time=0.20,
            t_end=0.30,
        )
        record = model.run_network(case, verbose=False)
        times = np.asarray(record["frames_t"], dtype=float)
        liquid = np.asarray(record["frames_alt"], dtype=float)
        x = np.asarray(record["xt"], dtype=float)
        release = x > case.x_valve

        self.assertTrue(np.all(liquid[0, release] == 0.0))
        index = int(np.argmin(np.abs(times - 0.20)))
        profile = liquid[index, release]
        wet = np.flatnonzero(profile > 1.0e-12)
        self.assertGreater(wet.size, 0)
        self.assertLess(wet.size, int(np.count_nonzero(release)))
        self.assertTrue(np.all(profile[wet[-1] + 1 :] == 0.0))
        self.assertEqual(float(record["dbg_created"]["t_floor"]), 0.0)
        self.assertTrue(
            np.all(np.asarray(record["top_liquid_outflow"], dtype=float) == 0.0)
        )
        self.assertEqual(float(record["top_liquid_outflow_volume"]), 0.0)

        gas_mass = np.asarray(record["tun_gas_mass"], dtype=float)
        relative_drift = abs(gas_mass[-1] / gas_mass[0] - 1.0)
        self.assertLess(relative_drift, 1.0e-4)

    def test_open_top_books_excess_as_control_surface_outflow(self) -> None:
        model = load_model()
        retained, outflow = model._book_top_liquid_outflow(1.25, 1.00)
        self.assertEqual(retained, 1.00)
        self.assertEqual(outflow, 0.25)
        retained, outflow = model._book_top_liquid_outflow(0.75, 1.00)
        self.assertEqual(retained, 0.75)
        self.assertEqual(outflow, 0.0)

    def test_circular_simple_wave_has_gate_trace_and_smooth_dry_toe(self) -> None:
        model = load_model()
        distance, area_fraction, velocity, mean_area = model._dry_bed_similarity()

        self.assertAlmostEqual(float(distance[0]), 0.0, places=12)
        self.assertAlmostEqual(float(distance[-1]), 1.0, places=12)
        self.assertAlmostEqual(float(area_fraction[0]), 0.4709576636918253, places=10)
        self.assertEqual(float(area_fraction[-1]), 0.0)
        self.assertTrue(np.all(np.diff(area_fraction) <= 1.0e-12))
        self.assertTrue(np.all(np.diff(velocity) >= -1.0e-12))
        self.assertGreater(mean_area, 0.0)
        self.assertLess(mean_area, float(area_fraction[0]))

    def test_reflected_front_projection_is_conservative_and_moves_off_cap(self) -> None:
        model = load_model()
        diameter = 0.050
        full_area = 0.25 * np.pi * diameter**2
        dx = 0.02
        centres = (np.arange(20, dtype=float) + 0.5) * dx
        cap_x = 0.40
        area = np.full(20, full_area)
        # A supported pocket with 30% void from x=0.08 to 0.24 m.  Because
        # 0.30 < reported k_a=0.41, the reflected-water front must detach from
        # the cap when the same gas volume is projected at k_a.
        area[4:12] = 0.70 * full_area
        discharge = 0.12 * area
        rho_air = model.P_ATM / (model.R_GAS * model.T_GAS)
        gas_mass = np.zeros(20)
        gas_mass[4:12] = rho_air * (full_area - area[4:12]) * dx

        gas_before = float(np.sum(gas_mass))
        void_before = float(np.sum(np.maximum(full_area - area, 0.0)) * dx)
        liquid_before = float(np.sum(area) * dx)
        discharge_before = float(np.sum(discharge) * dx)
        area_new, q_new, gas_new, nose_x, tail_x = model._project_reflected_pocket(
            area.copy(),
            discharge.copy(),
            gas_mass.copy(),
            centres,
            dx=dx,
            full_area=full_area,
            cap_x=cap_x,
            body_threshold=0.05,
        )

        self.assertTrue(np.isfinite(nose_x))
        self.assertLess(tail_x, cap_x)
        self.assertAlmostEqual(float(np.sum(gas_new)), gas_before, places=13)
        self.assertAlmostEqual(
            float(np.sum(np.maximum(full_area - area_new, 0.0)) * dx),
            void_before,
            places=13,
        )
        self.assertAlmostEqual(float(np.sum(area_new) * dx), liquid_before, places=13)
        self.assertAlmostEqual(float(np.sum(q_new) * dx), discharge_before, places=13)


if __name__ == "__main__":
    unittest.main()
