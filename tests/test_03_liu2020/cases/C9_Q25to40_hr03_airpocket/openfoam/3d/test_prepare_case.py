#!/usr/bin/env python3
"""Focused regression tests for C9 case generation."""

from __future__ import annotations

import json
import unittest

import prepare_case


class PerfectFluidEosTests(unittest.TestCase):
    def test_reference_density_and_wave_speed_are_preserved(self) -> None:
        density = 998.2
        pressure = 101325.0
        temperature = 293.15
        wave_speed = 305.0
        bulk_modulus = density * wave_speed**2

        fluid_constant, rho0, generated_speed = (
            prepare_case.perfect_fluid_eos_parameters(
                bulk_modulus,
                density,
                pressure,
                temperature,
            )
        )

        self.assertAlmostEqual(generated_speed, wave_speed)
        self.assertAlmostEqual(
            rho0 + pressure / (fluid_constant * temperature),
            density,
        )
        self.assertAlmostEqual(
            density * fluid_constant * temperature,
            bulk_modulus,
        )

    def test_invalid_bulk_modulus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            prepare_case.perfect_fluid_eos_parameters(
                0.0,
                998.2,
                101325.0,
                293.15,
            )


class PocketBodyTracerTests(unittest.TestCase):
    def test_tracer_excludes_the_thin_crown_layer(self) -> None:
        with prepare_case.PARAM_FILE.open(encoding="utf-8") as stream:
            parameters = json.load(stream)
        model = parameters["model"]
        pocket = model["pocket_profiles"]["base"]

        dictionary = prepare_case.generate_set_fields(
            parameters["paper"],
            model,
            pocket,
        )

        self.assertEqual(
            dictionary.count("volScalarFieldValue pocketBodyTracer 1"),
            1,
        )
        self.assertIn("volScalarFieldValue pocketBodyTracer 0", dictionary)
        tracer_index = dictionary.index(
            "volScalarFieldValue pocketBodyTracer 1"
        )
        containing_region = dictionary[
            dictionary.rfind("boxToCell", 0, tracer_index) : tracer_index
        ]
        self.assertIn(
            f"box ({pocket['tail_x_m']:.8g}",
            containing_region,
        )
        self.assertIn(
            f"({pocket['body_nose_x_m']:.8g} 0.12",
            containing_region,
        )


if __name__ == "__main__":
    unittest.main()
