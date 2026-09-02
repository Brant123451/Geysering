#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


RECOVERY_ROOT = Path(__file__).resolve().parent / "recovery_total_pressure_v1"
if str(RECOVERY_ROOT) in sys.path:
    sys.path.remove(str(RECOVERY_ROOT))
sys.path.insert(0, str(RECOVERY_ROOT))

import apply_total_pressure_profiles as converter
import audit_initial_fields as initial_audit
from test_apply_total_pressure_profiles import _make_case


def _stamp_exact_v2512_source_evidence(case: Path) -> None:
    audit_path = case / "total_pressure_profile_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_sources = {
        converter.PRGH_PRESSURE_SOURCE: converter.PRGH_PRESSURE_SOURCE_SHA256,
        converter.PRGH_PRESSURE_HEADER: converter.PRGH_PRESSURE_HEADER_SHA256,
        converter.PRGH_TOTAL_PRESSURE_SOURCE:
            converter.PRGH_TOTAL_PRESSURE_SOURCE_SHA256,
        converter.PRGH_TOTAL_PRESSURE_HEADER:
            converter.PRGH_TOTAL_PRESSURE_HEADER_SHA256,
    }
    audit["openfoam_source_verification"] = {
        "source_verified": True,
        "status": "verified",
        "required": True,
        "files": {
            source: {
                "expected_sha256": digest,
                "actual_sha256": digest,
                "verified": True,
            }
            for source, digest in expected_sources.items()
        },
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class PressureRoleInitialFieldGateTests(unittest.TestCase):
    def test_converted_v2_mixed_role_field_and_deterministic_audit_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, _ = _make_case(Path(temp))
            converter.apply_total_pressure_profiles(case)
            _stamp_exact_v2512_source_evidence(case)
            text = (case / "0" / "p_rgh").read_text(encoding="utf-8")

            report = initial_audit.total_pressure_profile_gate(case, text)

            self.assertTrue(report["passed"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                report["patches"]["waterInlet"]["type"],
                "prghTotalPressure",
            )
            self.assertEqual(
                report["patches"]["waterInlet"]["pressure_entry"], "p0"
            )
            for patch in (
                "waterOutlet",
                "ambientFloor",
                "ambientSides",
                "ambientTop",
            ):
                self.assertEqual(report["patches"][patch]["type"], "prghPressure")
                self.assertEqual(report["patches"][patch]["pressure_entry"], "p")
                self.assertTrue(report["patches"][patch]["passed"])
            self.assertTrue(report["patches"]["waterInlet"]["passed"])
            self.assertTrue(
                report["checks"]["openfoam_v2512_mixed_pressure_role_translation"]
            )
            self.assertTrue(
                report["checks"]["local_openfoam_v2512_pressure_sources_verified"]
            )

    def test_post_audit_field_mutation_fails_hash_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, _ = _make_case(Path(temp))
            converter.apply_total_pressure_profiles(case)
            field = case / "0" / "p_rgh"
            field.write_text(
                field.read_text(encoding="utf-8") + "\n// mutation after audit\n",
                encoding="utf-8",
                newline="\n",
            )

            report = initial_audit.total_pressure_profile_gate(
                case, field.read_text(encoding="utf-8")
            )

            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["field_sha256_matches_audit"])

    def test_tampered_converter_constant_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            case, _ = _make_case(Path(temp))
            converter.apply_total_pressure_profiles(case)
            audit_path = case / "total_pressure_profile_audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["constants"]["rho_water_kg_m3"] = 1000.0
            audit_path.write_text(json.dumps(audit), encoding="utf-8", newline="\n")

            field = case / "0" / "p_rgh"
            report = initial_audit.total_pressure_profile_gate(
                case, field.read_text(encoding="utf-8")
            )

            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["converter_constants_exact"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
