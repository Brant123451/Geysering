#!/usr/bin/env python3
"""Regression tests for Case-B evidence and resume policies."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from postprocess import (
    apply_acceptance,
    is_baseline_full,
    is_baseline_full_physics,
    is_canonical_hold,
    should_update_hold_evidence,
)
from resume_manifest import ENVIRONMENT_KEYS, read_manifest, shell_exports


HERE = Path(__file__).resolve().parent


def baseline_manifest(mesh: str = "base") -> dict:
    return {
        "stage": "full",
        "end_time_s": 10.5,
        "mesh_preset": mesh,
        "valve_mode": "opening",
        "valve_open_time_s": 0.25,
        "valve_seal_speed_m_per_s": 1.0,
        "initial_air_head_m": 0.610,
        "gas_equation_of_state": "perfectGas",
        "max_co": 0.30,
        "max_alpha_co": 0.20,
        "max_delta_t_s": 0.00025,
        "field_write_interval_s": 0.10,
        "c_alpha": 1.0,
        "alpha_smooth_curvature_iterations": 2,
    }


class BaselinePolicyTests(unittest.TestCase):
    def test_only_base_mesh_is_canonical(self) -> None:
        base = baseline_manifest("base")
        refined = baseline_manifest("refined")
        self.assertTrue(is_baseline_full_physics(base))
        self.assertTrue(is_baseline_full_physics(refined))
        self.assertTrue(is_baseline_full(base))
        self.assertFalse(is_baseline_full(refined))

    def test_numerical_sensitivity_is_not_baseline(self) -> None:
        manifest = baseline_manifest()
        manifest["valve_seal_speed_m_per_s"] = 0.5
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["max_delta_t_s"] = 0.0005
        self.assertFalse(is_baseline_full_physics(manifest))
        manifest = baseline_manifest()
        manifest["alpha_smooth_curvature_iterations"] = 0
        self.assertFalse(is_baseline_full_physics(manifest))

    def test_acceptance_can_only_complete_base_mesh(self) -> None:
        for mesh, expected in (("base", "complete"), ("refined", "incomplete")):
            metrics = {
                "run_configuration": baseline_manifest(mesh),
                "mesh_sensitivity": {"available": True},
                "openfoam_3d": {
                    "end_Tstar": 6.1,
                    "geyser": True,
                    "liquid_mass_error_pct_max_abs": 0.1,
                    "gas_mass_error_pct_max_abs": 0.1,
                    "geyser_height_censored_by_domain": False,
                },
            }
            apply_acceptance(metrics, hold_passed=True)
            self.assertEqual(
                metrics["openfoam_3d"]["completion_status"], expected
            )


class HoldEvidenceTests(unittest.TestCase):
    def test_canonical_hold_requires_baseline_controls(self) -> None:
        manifest = baseline_manifest()
        manifest.update(stage="hold", valve_mode="closed", end_time_s=1.0)
        self.assertTrue(is_canonical_hold(manifest))
        manifest["mesh_preset"] = "refined"
        self.assertFalse(is_canonical_hold(manifest))

    def test_passed_hold_cannot_be_downgraded(self) -> None:
        existing = {"hold_test": {"duration_s": 1.0, "passed": True}}
        candidate = {"hold_test": {"duration_s": 2.0, "passed": False}}
        self.assertFalse(should_update_hold_evidence(existing, candidate))

    def test_longer_failed_diagnostic_replaces_shorter_only(self) -> None:
        existing = {"hold_test": {"duration_s": 0.01, "passed": False}}
        shorter = {"hold_test": {"duration_s": 0.001, "passed": False}}
        longer = {"hold_test": {"duration_s": 1.0, "passed": False}}
        self.assertFalse(should_update_hold_evidence(existing, shorter))
        self.assertTrue(should_update_hold_evidence(existing, longer))


class ResumeManifestTests(unittest.TestCase):
    def test_all_materialised_controls_are_exported(self) -> None:
        manifest = baseline_manifest()
        manifest["stage"] = "hold"
        manifest["valve_mode"] = "closed"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            restored = read_manifest(path)
        exports = shell_exports(restored)
        for environment_key in ENVIRONMENT_KEYS.values():
            self.assertIn(f"export {environment_key}=", exports)

    def test_missing_control_is_rejected(self) -> None:
        manifest = baseline_manifest()
        del manifest["max_co"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing controls"):
                read_manifest(path)


class InitialFieldPolicyTests(unittest.TestCase):
    def test_cut_cells_use_mixture_density_for_reduced_pressure(self) -> None:
        text = (HERE / "system" / "setExprFieldsDict").read_text()
        self.assertIn("mixtureReducedPressure", text)
        self.assertIn("(1 - alpha.water)*(p/(287.058*293.15))", text)
        self.assertNotIn("reducedWaterPressure", text)
        self.assertNotIn("reducedAirPressure", text)


if __name__ == "__main__":
    unittest.main()
