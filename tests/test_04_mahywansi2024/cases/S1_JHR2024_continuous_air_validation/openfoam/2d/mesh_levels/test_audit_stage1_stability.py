#!/usr/bin/env python3
"""Synthetic, OpenFOAM-free tests for audit_stage1_stability.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import audit_stage1_stability as audit


def _field_text(dimensions: str, inlet_value: str, outlet_value: str) -> str:
    return f"""
FoamFile
{{
    format ascii;
}}
dimensions {dimensions};
internalField uniform 0;
boundaryField
{{
    waterInlet
    {{
        type calculated;
        value {inlet_value};
    }}
    waterOutlet {{
        type calculated;
        value {outlet_value};
    }}
    walls
    {{
        type calculated;
        value nonuniform List<scalar>
        4
        (
            100
            101
            102
            103
        );
    }}
}}
"""


def _probe_text(rows: list[str], vector: bool = False) -> str:
    del vector
    return "\n".join(
        [
            "# Probe 0 (0 0 0)",
            "# Probe 1 (1 0 0)",
            "# Time 0 1",
            *rows,
            "",
        ]
    )


class ScalarPatchParserTests(unittest.TestCase):
    def test_multiline_nonuniform_and_uniform_density(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            time_dir = Path(temporary) / "1.5"
            time_dir.mkdir()
            (time_dir / "phi").write_text(
                _field_text(
                    "[0 3 -1 0 0 0 0]",
                    "nonuniform List<scalar> 2 (-1e-5 -2e-5)",
                    "nonuniform List<scalar>\n2\n(1.4e-5\n1.6e-5)",
                ),
                encoding="utf-8",
            )
            (time_dir / "rho").write_text(
                _field_text(
                    "[1 -3 0 0 0 0 0]",
                    "uniform 1000",
                    "nonuniform List<scalar> 2 (999 1001)",
                ),
                encoding="utf-8",
            )
            snapshot = audit.parse_flux_snapshot(1.5, time_dir)
            self.assertAlmostEqual(snapshot["qin_m3_per_s"], 3.0e-5)
            self.assertAlmostEqual(snapshot["qout_m3_per_s"], 3.0e-5)
            self.assertAlmostEqual(snapshot["mdot_in_kg_per_s"], 0.03)
            self.assertAlmostEqual(
                snapshot["mdot_out_kg_per_s"], 999 * 1.4e-5 + 1001 * 1.6e-5
            )

    def test_dimension_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            time_dir = Path(temporary) / "1"
            time_dir.mkdir()
            (time_dir / "phi").write_text(
                _field_text(
                    "[1 0 -1 0 0 0 0]",
                    "uniform -1e-5",
                    "uniform 1e-5",
                ),
                encoding="utf-8",
            )
            (time_dir / "rho").write_text(
                _field_text(
                    "[1 -3 0 0 0 0 0]",
                    "uniform 1000",
                    "uniform 1000",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(audit.AuditInputError):
                audit.parse_flux_snapshot(1.0, time_dir)


class ProbeParserTests(unittest.TestCase):
    def test_restart_segments_merge_and_later_duplicate_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary)
            first = case / "postProcessing" / "probesJHR" / "0"
            restart = case / "postProcessing" / "probesJHR" / "0.5"
            first.mkdir(parents=True)
            restart.mkdir(parents=True)
            (first / "p").write_text(
                _probe_text(["0 1 2", "0.5 3 4"]), encoding="utf-8"
            )
            (restart / "p").write_text(
                _probe_text(["0.5 30 40", "1 31 41"]), encoding="utf-8"
            )
            (first / "U").write_text(
                _probe_text(
                    ["0 (1 0 0) (2 0 0)", "0.5 (3 0 0) (4 0 0)"],
                    vector=True,
                ),
                encoding="utf-8",
            )
            (restart / "U").write_text(
                _probe_text(
                    ["0.5 (30 0 0) (40 0 0)", "1 (31 0 0) (41 0 0)"],
                    vector=True,
                ),
                encoding="utf-8",
            )

            count, pressure, segments = audit.merge_probe_segments(
                case, "p", vector=False
            )
            self.assertEqual(count, 2)
            self.assertEqual(segments, ["0", "0.5"])
            self.assertEqual([row[0] for row in pressure], [0.0, 0.5, 1.0])
            self.assertEqual(pressure[1][1], [30.0, 40.0])

            _, velocity, _ = audit.merge_probe_segments(case, "U", vector=True)
            self.assertEqual(velocity[1][1], [[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]])


class GateAndMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = json.loads(audit.GATE_PATH.read_text(encoding="utf-8"))

    def test_current_output_plan_is_sufficient_without_running_openfoam(self) -> None:
        for level, meta in audit.LEVELS.items():
            with self.subTest(level=level):
                report = audit.inspect_output_capability(meta, self.gate)
                self.assertTrue(report["sufficient_for_registered_quasi_steady_audit"])
                self.assertEqual(report["runtime_saved_field_interval_s"], 0.1)
                self.assertEqual(report["runtime_purge_write"], 0)
                self.assertLessEqual(report["nominal_max_probe_spacing_s"], 0.02)

    def test_constant_synthetic_terminal_window_passes_all_metrics(self) -> None:
        probe_times = [12.0 + index * 0.02 for index in range(201)]
        pressure = [
            (time, [107000.0 + probe * 10.0 for probe in range(6)])
            for time in probe_times
        ]
        pressure_rgh = [
            (time, [101500.0 + probe * 5.0 for probe in range(6)])
            for time in probe_times
        ]
        velocity = [
            (time, [[0.05, 0.0, 0.0] for _ in range(6)])
            for time in probe_times
        ]
        flux = [
            {
                "time_s": 12.0 + index * 0.1,
                "qin_m3_per_s": 5.0e-5,
                "qout_m3_per_s": 5.0e-5,
                "mdot_in_kg_per_s": 0.05,
                "mdot_out_kg_per_s": 0.05,
            }
            for index in range(41)
        ]
        metrics, checks = audit.calculate_metrics(
            self.gate,
            12.0,
            16.0,
            {"p": pressure, "p_rgh": pressure_rgh, "U": velocity},
            flux,
        )
        self.assertIn("pressure_probes", metrics)
        self.assertTrue(checks)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_next_endpoint_never_promotes_short_run(self) -> None:
        self.assertEqual(
            audit.recommended_next_endpoint(self.gate, latest=0.5, stable=False), 16.0
        )
        self.assertEqual(
            audit.recommended_next_endpoint(self.gate, latest=16.0, stable=False), 20.0
        )
        self.assertIsNone(
            audit.recommended_next_endpoint(self.gate, latest=16.0, stable=True)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
