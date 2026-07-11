#!/usr/bin/env python3
"""Focused regression tests for C9 three-dimensional post-processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import postprocess_openfoam as post


class DominantGasComponentTests(unittest.TestCase):
    def test_detached_downstream_gas_is_not_the_main_body_front(self) -> None:
        x = np.arange(-5.0, 1.0)
        alpha_water = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        front, span, furthest = post.dominant_gas_component(x, alpha_water)

        self.assertEqual(front, -2.0)
        self.assertEqual(span, 2.0)
        self.assertEqual(furthest, 0.0)

    def test_contiguous_main_body_can_reach_the_last_probe(self) -> None:
        x = np.arange(-5.0, 1.0)
        alpha_water = np.array([1.0, 0.2, 0.1, 0.0, 0.3, 0.4])

        front, span, furthest = post.dominant_gas_component(x, alpha_water)

        self.assertEqual(front, 0.0)
        self.assertEqual(span, 4.0)
        self.assertEqual(furthest, 0.0)

    def test_no_gas_returns_nan_fronts(self) -> None:
        front, span, furthest = post.dominant_gas_component(
            np.array([-1.0, 0.0]), np.array([1.0, 1.0])
        )

        self.assertTrue(np.isnan(front))
        self.assertEqual(span, 0.0)
        self.assertTrue(np.isnan(furthest))


class ProbeParsingTests(unittest.TestCase):
    def test_scalar_probe_coordinates_come_from_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            post_dir = Path(directory)
            output = post_dir / "deep" / "0" / "alpha.water"
            output.parent.mkdir(parents=True)
            output.write_text(
                "\n".join(
                    [
                        "# Probe 0 (-1.2 0 0.4)",
                        "# Probe 1 (-0.17 0 0.39)",
                        "# Time",
                        "0.0 0.0 1.0",
                        "0.1 0.2 0.8",
                    ]
                ),
                encoding="utf-8",
            )

            times, values, locations = post.parse_probe_scalar_with_locations(
                post_dir, "deep", "alpha.water"
            )

        np.testing.assert_allclose(times, [0.0, 0.1])
        np.testing.assert_allclose(values, [[0.0, 1.0], [0.2, 0.8]])
        np.testing.assert_allclose(locations[:, 0], [-1.2, -0.17])

    def test_scalar_probe_width_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            post_dir = Path(directory)
            output = post_dir / "deep" / "0" / "alpha.water"
            output.parent.mkdir(parents=True)
            output.write_text(
                "\n".join(
                    [
                        "# Probe 0 (-1 0 0.4)",
                        "# Probe 1 (0 0 0.4)",
                        "0.0 0.0 1.0",
                        "0.1 0.2",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected 2"):
                post.parse_probe_scalar_with_locations(
                    post_dir, "deep", "alpha.water"
                )

    def test_restart_header_rounding_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            post_dir = Path(directory)
            first = post_dir / "deep" / "0" / "alpha.water"
            restart = post_dir / "deep" / "0.25" / "alpha.water"
            first.parent.mkdir(parents=True)
            restart.parent.mkdir(parents=True)
            first.write_text(
                "# Probe 0 (0.15 0 2.54946)\n0.0 1.0\n",
                encoding="utf-8",
            )
            restart.write_text(
                "# Probe 0 (0.15 0 2.549455)\n0.25 0.8\n",
                encoding="utf-8",
            )

            times, values, locations = post.parse_probe_scalar_with_locations(
                post_dir, "deep", "alpha.water"
            )

        np.testing.assert_allclose(times, [0.0, 0.25])
        np.testing.assert_allclose(values[:, 0], [1.0, 0.8])
        self.assertAlmostEqual(locations[0, 2], 2.54946)

    def test_sustained_state_does_not_bridge_output_gap(self) -> None:
        time = np.array([0.00, 0.01, 0.02, 0.20, 0.21, 0.22])
        active = np.ones(len(time), dtype=bool)

        detected = post.first_sustained_time(
            time,
            active,
            minimum_duration=0.05,
            maximum_gap=0.025,
        )

        self.assertIsNone(detected)


class LogParsingTests(unittest.TestCase):
    def test_strict_mesh_pass_reports_zero_concave_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "log.checkMesh").write_text(
                "cells: 873032\nMesh OK.\n", encoding="utf-8"
            )
            (case / "log.checkMesh.all").write_text(
                "\n".join(
                    [
                        "cells: 873032",
                        "Max aspect ratio = 6.47 OK.",
                        "Min volume = 2.57e-10. Max volume = 1.",
                        "Mesh non-orthogonality Max: 34.8 average: 3.9",
                        "Max skewness = 1.23 OK.",
                        "*There are 12 faces with concave angles between consecutive edges. "
                        "Max concave angle = 23.25 degrees.",
                        "Concave cell check OK.",
                        "Mesh OK.",
                    ]
                ),
                encoding="utf-8",
            )

            quality = post.parse_mesh_quality(case)

        self.assertTrue(quality["strict_check_run"])
        self.assertTrue(quality["all_geometry_passed"])
        self.assertEqual(quality["concave_cells"], 0)
        self.assertEqual(quality["concave_faces"], 12)
        self.assertAlmostEqual(quality["max_concave_face_angle_deg"], 23.25)

    def test_limiter_peak_keeps_stage_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "log.smoke").write_text(
                "\n".join(
                    [
                        "Time = 0.4",
                        "limitVelocity limitU Limited 2 (0.1%) of cells, "
                        "1 (0.2%) of faces, with max limit 12",
                        "Time = 0.5",
                        "limitVelocity limitU Limited 5 (0.3%) of cells, "
                        "3 (0.4%) of faces, with max limit 12",
                        "Time = 0.55",
                        "limitVelocity limitU Limited 4 (0.25%) of cells, "
                        "8 (1.2%) of faces, with max limit 12",
                    ]
                ),
                encoding="utf-8",
            )

            numerics = post.parse_numerics(case, paper_time_offset=0.25)

        self.assertEqual(numerics["maximum_limited_cells"], 5)
        self.assertEqual(numerics["maximum_limited_faces"], 8)
        self.assertAlmostEqual(numerics["maximum_limited_face_percent"], 1.2)
        self.assertEqual(numerics["maximum_limited_stage"], "smoke")
        self.assertAlmostEqual(numerics["maximum_limited_paper_time_s"], 0.25)
        self.assertAlmostEqual(
            numerics["limiter_by_stage"]["smoke"][
                "first_activation_paper_time_s"
            ],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
